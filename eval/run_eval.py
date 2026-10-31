"""Evaluation harness + compute-allocation ablation (the Snell et al. experiment).

For each chapter, the planner runs ONCE; every policy then writes from the same
outline, so differences come only from how test-time compute is spent:

  no_critic : 1 draft per segment, no verifier          (baseline)
  single    : 1 draft per segment, critic only scores    (N=1)
  uniform   : budget split evenly, best-of-N by critic   (classic best-of-N)
  adaptive  : difficulty-aware + verifier-guided realloc (ours)

Reported per policy: writer calls, LLM calls/tokens, wall time, an independent
judge's scores (different prompt; optionally a different model via EVAL_JUDGE_*),
and the claim-support rate of the chosen scripts. With --audio, the voice stage
also runs and reports ASR WER of first takes vs kept takes (self-check effect).

Usage:
  python eval/run_eval.py --chapters chem-reactions,crop-production --length short
  python eval/run_eval.py --pdf path/a.pdf --pdf path/b.pdf --audio
Results go to eval/results/<timestamp>/{results.json,results.md}.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.agent.claims import check_claims
from backend.agent.context import AgentContext
from backend.agent.orchestrator import write_segments
from backend.agent.planner import plan
from backend.agent.search import best_candidate
from backend.agent.voice import synthesize_version
from backend.agent.writer import script_text
from backend.clients.llm import chat_json
from backend.clients.usage import current_metrics
from backend.config import get_settings
from backend.ingest import ingest_pdf, ingest_text, select_chunks
from backend.prompts import EVAL_JUDGE, PROMPT_VERSION
from backend.schemas import Episode, EpisodeSettings, Metrics, Segment, SegmentVersion

POLICIES = ["no_critic", "single", "uniform", "adaptive"]
JUDGE_AXES = ["faithfulness", "naturalness", "engagement", "pedagogy"]


def judge_endpoint() -> tuple[str, str, str] | None:
    url = os.environ.get("EVAL_JUDGE_BASE_URL")
    return (url, os.environ.get("EVAL_JUDGE_API_KEY", ""), os.environ.get("EVAL_JUDGE_MODEL", "")) if url else None


async def load_sources(args) -> list:
    s = get_settings()
    srcs = []
    samples = json.loads((ROOT / "samples" / "samples.json").read_text())
    for sid in [x for x in (args.chapters or "").split(",") if x]:
        meta = next(m for m in samples if m["id"] == sid)
        srcs.append(ingest_text((ROOT / "samples" / meta["file"]).read_text(), s.max_input_words, origin="sample", title=meta["title"]))
    for p in args.pdf or []:
        srcs.append(await ingest_pdf(Path(p).read_bytes(), s.max_input_words))
    if not srcs:
        for meta in samples:
            srcs.append(ingest_text((ROOT / "samples" / meta["file"]).read_text(), s.max_input_words, origin="sample", title=meta["title"]))
    return srcs


def fresh_copy(segments: list[Segment]) -> list[Segment]:
    return [s.model_copy(deep=True, update={"candidates": [], "reflections": [], "versions": [], "budget": 0, "status": "pending"})
            for s in segments]


async def judge(ctx: AgentContext, seg: Segment, turns) -> dict:
    st = ctx.episode.settings
    msg = EVAL_JUDGE.format(passages=ctx.passages_text(seg.chunk_ids, 1200), script=script_text(turns, (st.host_name, st.expert_name)))
    try:
        r = await chat_json([{"role": "user", "content": msg}], task="judge", temperature=0.0, max_tokens=300, endpoint=judge_endpoint())
        return {k: float(r[k]) for k in JUDGE_AXES if isinstance(r.get(k), (int, float))}
    except Exception as e:
        print("  judge failed:", e)
        return {}


async def run_policy(src, chunks, outline: list[Segment], policy: str, length: str, audio: bool, out_dir: Path) -> dict:
    ep = Episode(id=f"eval_{policy}", source_id=src.id, title=src.title, settings=EpisodeSettings(length=length, policy=policy))
    ep.segments = fresh_copy(outline)
    metrics = Metrics()
    ep.metrics = metrics
    token = current_metrics.set(metrics)
    ctx = AgentContext(episode=ep, source=src, chunks=chunks)
    t0 = time.time()
    try:
        await write_segments(ctx, ep.segments, ep.segments)
        write_s = time.time() - t0
        write_calls = metrics.llm_calls
        judged, critic, claims_ok, claims_n = [], [], 0, 0
        first_wers, kept_wers = [], []
        for seg in ep.segments:
            best = best_candidate(seg)
            if best is None:
                continue
            if best.scores:
                critic.append(best.scores.overall)
            # (critic scores are not comparable for no_critic, where there are none)
            j = await judge(ctx, seg, best.turns)
            if j:
                judged.append(j)
            cl = await check_claims(ctx, seg, best.turns)
            claims_ok += sum(c.supported for c in cl)
            claims_n += len(cl)
            if audio:
                ver = SegmentVersion(version=1, turns=[t.model_copy() for t in best.turns])
                path = out_dir / f"{src.id}_{policy}_{seg.id}.wav"
                await synthesize_version(ctx, seg, ver, path)
                first_wers += [t.first_wer for t in ver.turns if t.first_wer is not None]
                kept_wers += [t.wer for t in ver.turns if t.wer is not None]
    finally:
        current_metrics.reset(token)
    res = {
        "policy": policy,
        "writer_calls": sum(s.budget for s in ep.segments),
        "llm_calls_writing": write_calls,
        "tokens": metrics.prompt_tokens + metrics.completion_tokens,
        "write_seconds": round(write_s, 1),
        "critic_best": round(statistics.mean(critic), 2) if critic else None,
        "claim_support": round(claims_ok / claims_n, 3) if claims_n else None,
        "segments": len(ep.segments),
    }
    for k in JUDGE_AXES:
        vals = [j[k] for j in judged if k in j]
        res[f"judge_{k}"] = round(statistics.mean(vals), 2) if vals else None
    vals = [statistics.mean(j.values()) for j in judged if j]
    res["judge_mean"] = round(statistics.mean(vals), 2) if vals else None
    if audio:
        res["wer_first_take"] = round(statistics.mean(first_wers), 3) if first_wers else None
        res["wer_after_selfcheck"] = round(statistics.mean(kept_wers), 3) if kept_wers else None
    res["difficulty_budget"] = [(s.difficulty, s.budget) for s in ep.segments]
    return res


def to_markdown(all_rows: list[dict], meta: dict) -> str:
    cols = ["policy", "writer_calls", "tokens", "write_seconds", "critic_best", "judge_mean"] + [f"judge_{k}" for k in JUDGE_AXES] + ["claim_support"]
    if any("wer_first_take" in r for r in all_rows):
        cols += ["wer_first_take", "wer_after_selfcheck"]
    out = [f"# Eval results ({meta['time']})", "", f"prompts {meta['prompt_version']} · length {meta['length']} · judge: {meta['judge']} · mock: {meta['mock']}", ""]
    chapters = list(dict.fromkeys(r["chapter"] for r in all_rows))
    for ch in chapters:
        out += [f"## {ch}", "", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        for r in [r for r in all_rows if r["chapter"] == ch]:
            out.append("| " + " | ".join("" if r.get(c) is None else str(r.get(c)) for c in cols) + " |")
        out.append("")
    out += ["## Mean over chapters", "", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for p in dict.fromkeys(r["policy"] for r in all_rows):
        rows = [r for r in all_rows if r["policy"] == p]
        cells = [p]
        for c in cols[1:]:
            vals = [r[c] for r in rows if isinstance(r.get(c), (int, float))]
            cells.append(str(round(statistics.mean(vals), 3)) if vals else "")
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chapters", help="comma-separated sample ids (default: all samples)")
    ap.add_argument("--pdf", action="append", help="extra chapter PDFs")
    ap.add_argument("--policies", default=",".join(POLICIES))
    ap.add_argument("--length", default="short", choices=["short", "medium", "long"])
    ap.add_argument("--audio", action="store_true", help="also synthesise and measure ASR self-check WER")
    args = ap.parse_args()
    out_dir = ROOT / "eval" / "results" / time.strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for src in await load_sources(args):
        chunks, _ = select_chunks(src, [], get_settings().max_input_words)
        base = Episode(id="eval_plan", source_id=src.id, title=src.title, settings=EpisodeSettings(length=args.length))
        print(f"== {src.title}: planning once for all policies")
        outline = await plan(AgentContext(episode=base, source=src, chunks=chunks))
        for policy in args.policies.split(","):
            print(f"  -> {policy}")
            r = await run_policy(src, chunks, outline, policy, args.length, args.audio, out_dir)
            r["chapter"] = src.title
            rows.append(r)
            print("    ", {k: v for k, v in r.items() if k not in ("difficulty_budget",)})
    meta = {"time": time.strftime("%Y-%m-%d %H:%M"), "prompt_version": PROMPT_VERSION, "length": args.length,
            "judge": (judge_endpoint() or ("", "", "same model as writer"))[2] or "custom", "mock": get_settings().effective_mock()}
    (out_dir / "results.json").write_text(json.dumps({"meta": meta, "rows": rows}, indent=2))
    (out_dir / "results.md").write_text(to_markdown(rows, meta))
    print(f"\nWrote {out_dir / 'results.md'}")


if __name__ == "__main__":
    asyncio.run(main())
