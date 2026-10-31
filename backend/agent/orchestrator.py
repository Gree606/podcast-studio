"""The agent loop for a whole episode, and for single-segment regeneration.

Perceive  : ingest the chapter, explore it with tools (planner)
Plan      : outline + difficulty per segment + compute budget per segment
Act       : write drafts, revise, synthesise audio
Observe   : critic scores, claim checks, ASR self-check of the audio
...and iterate where an observation says the result isn't good enough.

Robustness rules: one failing segment never fails the episode; every stage
reports progress; all state is persisted so a restart can resume.
"""
from __future__ import annotations

import asyncio
import logging
import time
import traceback

import numpy as np

from ..audio.stitch import concat_segments, decode_audio, encode_mp3, to_wav_bytes
from ..clients import embed
from ..clients.base import ServiceError
from ..clients.usage import current_metrics
from ..config import get_settings
from ..ingest import select_chunks
from ..prompts import PROMPT_VERSION
from ..schemas import Claim, Episode, Segment, SegmentVersion
from ..store import store
from . import fast
from .allocation import initial_allocation, reallocate
from .claims import verify_and_fix
from .context import AgentContext
from .planner import plan
from .search import best_candidate, run_search
from .voice import prefetch, synthesize_version

log = logging.getLogger("podcast.orchestrator")


def make_context(ep: Episode) -> AgentContext:
    src = store.get_source(ep.source_id)
    if src is None:
        raise ServiceError("Storage", "lost the uploaded chapter; please upload it again")
    chunks, _ = select_chunks(src, ep.settings.section_ids, get_settings().max_input_words)
    ctx = AgentContext(episode=ep, source=src, chunks=chunks, on_change=lambda: store.save_episode(ep, force=False))
    if not fast_mode(ep) and embed.embed_ready():
        ctx.use_hybrid_search()
        ctx.emit("plan", "info", f"Retrieval: keyword (BM25) + meaning search ({get_settings().embed_model} embeddings, "
                 "SQLite vector store), fused by reciprocal rank.")
    return ctx


class _Timer:
    """Context manager adding the wall-clock time of a stage to ``ep.metrics.stage_seconds``."""

    def __init__(self, ep: Episode, stage: str):
        self.ep, self.stage = ep, stage

    def __enter__(self):
        self.t0 = time.time()

    def __exit__(self, *exc):
        self.ep.metrics.stage_seconds[self.stage] = round(self.ep.metrics.stage_seconds.get(self.stage, 0) + time.time() - self.t0, 1)


async def _finalize_segment(ctx: AgentContext, seg: Segment, segments: list[Segment], instruction: str | None = None) -> None:
    """Pick the best candidate, claim-check it, synthesise, and store it as a new version."""
    best = best_candidate(seg)
    if best is None:
        raise ServiceError("Writer", "could not produce a script for this segment")
    seg.status = "checking"
    turns, claims = await verify_and_fix(ctx, seg, segments, [t.model_copy() for t in best.turns])
    ver = SegmentVersion(version=fast.next_version_number(seg), instruction=instruction,
                         turns=turns, scores=best.scores, claims=claims)
    await _record(ctx, seg, ver)


async def _record(ctx: AgentContext, seg: Segment, ver: SegmentVersion) -> None:
    """Synthesise a checked script (TTS + ASR listen-back) and make it the segment's active version."""
    seg.status = "synthesizing"
    ctx.on_change()
    path = store.episode_dir(ctx.episode.id) / f"{seg.id}_v{ver.version}.wav"
    await synthesize_version(ctx, seg, ver, path)
    seg.versions.append(ver)
    seg.active_version = ver.version
    seg.status = "done"
    seg.error = None
    if ctx.episode.status == "running":  # listeners can start on this part while the rest is still recording
        ctx.emit("voice", "info", f"Part {seg.index + 1} is recorded and ready to play.", seg.id)


async def assemble(ctx: AgentContext) -> None:
    """Concatenate each segment's active version into the episode MP3 and compute absolute timestamps."""
    ep = ctx.episode
    sr = get_settings().sample_rate
    d = store.episode_dir(ep.id)
    audios: list[np.ndarray] = []
    used: list[Segment] = []
    for seg in ep.segments:
        v = seg.current
        if v and v.audio_file and (d / v.audio_file).exists():
            audios.append(decode_audio((d / v.audio_file).read_bytes(), sr))
            used.append(seg)
    if not audios:
        raise ServiceError("Audio", "no segment produced audio")
    audio, offsets = concat_segments(audios, sr)
    for seg, off in zip(used, offsets):
        for t in seg.current.turns:
            if t.seg_start is not None:
                t.start, t.end = round(off + t.seg_start, 3), round(off + (t.seg_end or t.seg_start), 3)
    rev = int(time.time() * 1000)
    try:
        data, name = await asyncio.to_thread(encode_mp3, audio, sr), f"episode_{rev}.mp3"
    except RuntimeError as e:  # never lose a finished episode over the encoder: serve WAV instead
        log.warning("mp3 encode failed, falling back to wav: %s", e)
        data, name = to_wav_bytes(audio, sr), f"episode_{rev}.wav"
    (d / name).write_bytes(data)
    old = ep.audio_file
    ep.audio_file = name
    ep.duration = round(len(audio) / sr, 1)
    if old and old != name and (d / old).exists():
        (d / old).unlink()


async def write_segments(ctx: AgentContext, segments: list[Segment], need_write: list[Segment]) -> None:
    """Spend the episode's writer budget across segments (Snell et al.-style allocation).

    Round 1 uses predicted difficulty; for the adaptive policy, leftover budget is then
    reallocated to segments the verifier still scores below the acceptance bar.
    """
    s = get_settings()
    policy = ctx.episode.settings.policy
    total_budget = max(len(segments), round(s.budget_per_segment * len(segments)))
    alloc = initial_allocation([x.difficulty for x in need_write], total_budget, policy, s.max_candidates_per_segment)
    ctx.emit("write", "decision", f"Compute plan ({policy}): total budget {total_budget} writer calls; "
             "first round " + ", ".join(f"{x.title[:24]}={a}" for x, a in zip(need_write, alloc)))
    done_count = 0

    async def search(seg: Segment, budget: int) -> int:
        try:
            return await run_search(ctx, seg, segments, budget, policy=policy, accept=s.accept_score)
        except Exception:  # a bug or outage in one segment must not sink the episode
            log.error("writing %s failed: %s", seg.id, traceback.format_exc())
            ctx.emit("write", "error", f"Writing '{seg.title}' failed unexpectedly.", seg.id)
            return budget

    async def first_round(seg: Segment, budget: int) -> int:
        nonlocal done_count
        seg.status = "writing"
        spent = await search(seg, budget)
        done_count += 1
        ctx.status("write", f"Writing: {done_count}/{len(need_write)} segments drafted", 0.15 + 0.45 * done_count / len(need_write))
        return spent

    spent = list(await asyncio.gather(*(first_round(x, a) for x, a in zip(need_write, alloc))))
    for x, sp in zip(need_write, spent):
        x.budget = sp
    remaining = total_budget - sum(spent)
    if policy != "adaptive" or remaining <= 0:
        return

    def best_score(x: Segment) -> float:
        b = best_candidate(x)
        return b.scores.overall if b and b.scores else 0.0

    bests = [best_score(x) for x in need_write]
    extra = reallocate(remaining, bests, [x.difficulty for x in need_write], spent, s.accept_score, s.max_candidates_per_segment)
    if not any(extra):
        ctx.emit("write", "decision", f"All segments met the bar; {remaining} call(s) of budget left unused.")
        return
    ctx.emit("write", "decision", f"Verifier-guided reallocation of {remaining} leftover call(s): " +
             ", ".join(f"{x.title[:24]} +{e} (best {b:.2f})" for x, e, b in zip(need_write, extra, bests) if e))
    ctx.status("write", "Improving the weakest segments", 0.6)
    chosen = [(x, e) for x, e in zip(need_write, extra) if e]
    more = await asyncio.gather(*(search(x, e) for x, e in chosen))
    for (x, _), sp in zip(chosen, more):
        x.budget += sp


def fast_mode(ep: Episode) -> bool:
    """The episode's own choice (made on the launch screen), else the server default."""
    return (ep.settings.mode or get_settings().pipeline_mode) != "agent"


async def _run_fast(ctx: AgentContext) -> None:
    """FAST pipeline, LLM call 1: write the whole episode (skipped when resuming with a script)."""
    ep = ctx.episode
    if ep.segments:
        return
    with _Timer(ep, "write"):
        ep.segments = await fast.write_episode(ctx)
    store.save_episode(ep)


async def run_episode(eid: str) -> None:
    ep = store.get_episode(eid)
    if ep is None:
        return
    token = current_metrics.set(ep.metrics)
    try:
        async with store.lock(eid):
            ep.status, ep.error = "running", None
            ctx = make_context(ep)
            ctx.emit("start", "info", f"Agent started (prompts {PROMPT_VERSION}, policy {ep.settings.policy}, "
                     f"{len(ctx.chunks)} passages in scope).")
            quick = fast_mode(ep)

            if quick:
                await _run_fast(ctx)
            # ---- Perceive + Plan ---------------------------------------------------
            elif not ep.segments:
                ctx.status("plan", "Reading the chapter and planning the episode", 0.02)
                with _Timer(ep, "plan"):
                    ep.segments = await plan(ctx)
                ctx.emit("plan", "decision", "Outline: " + " | ".join(f"{x.index + 1}. {x.title} (d{x.difficulty})" for x in ep.segments))
                store.save_episode(ep)
            segments = ep.segments
            todo = [x for x in segments if not (x.status == "done" and x.current)]

            # ---- Act + Observe: write with compute-optimal allocation ----------------
            need_write = [] if quick else [x for x in todo if best_candidate(x) is None]
            if need_write:
                with _Timer(ep, "write"):
                    await write_segments(ctx, segments, need_write)
                store.save_episode(ep)

            # ---- Verify claims + synthesise + ASR self-check -----------------------
            claims: dict[str, list[Claim]] = {}
            if quick and todo:
                # Record while the fact-check runs (part 1 first). Nothing is released to listeners until
                # the check is done; any line it changes misses the cache and is recorded again.
                for x in todo:
                    if x.candidates:
                        prefetch(ctx, x, x.candidates[-1].turns)
                ctx.emit("voice", "info", "Started recording the lines while the fact-check runs; "
                         "any line the check changes is re-recorded.")
                with _Timer(ep, "verify"):
                    claims = await fast.verify_segments(ctx, segments, only=todo)
                store.save_episode(ep)
            ctx.status("voice", "Recording the hosts" if quick else "Fact-checking and recording the hosts", 0.62)
            finished = 0
            with _Timer(ep, "voice"):
                async def finish(seg: Segment) -> None:
                    nonlocal finished
                    try:
                        if quick:
                            await _record(ctx, seg, fast.new_version(seg, claims.get(seg.id, [])))
                        else:
                            await _finalize_segment(ctx, seg, segments)
                    except Exception as e:  # isolate failures per segment
                        seg.status = "failed"
                        seg.error = getattr(e, "user_message", None) or "Something went wrong while making this segment."
                        ctx.emit("voice", "error", f"Segment '{seg.title}' failed: {seg.error}", seg.id)
                        log.error("segment %s failed: %s", seg.id, traceback.format_exc())
                    finished += 1
                    ctx.status("voice", f"Recording: {finished}/{len(todo)} segments", 0.62 + 0.33 * finished / max(1, len(todo)))

                await asyncio.gather(*(finish(x) for x in todo))

            # ---- Assemble ---------------------------------------------------------
            ctx.status("assemble", "Mixing the final episode", 0.96)
            with _Timer(ep, "assemble"):
                await assemble(ctx)
            failed = [x for x in segments if x.status == "failed"]
            ep.status = "done"
            ep.progress = 1.0
            ep.stage = "done"
            ep.message = "Ready" if not failed else f"Ready, but {len(failed)} segment(s) failed - use Regenerate on them."
            ctx.emit("done", "info", f"Episode ready: {ep.duration:.0f}s, {ep.metrics.llm_calls} LLM calls, "
                     f"{ep.metrics.prompt_tokens + ep.metrics.completion_tokens} tokens.")
    except Exception as e:
        ep.status = "failed"
        ep.error = getattr(e, "user_message", None) and f"{e.service} {e.user_message}." or "Something went wrong while making the episode."
        ep.message = ep.error + " You can press Resume to try again."
        log.error("episode %s failed: %s", eid, traceback.format_exc())
    finally:
        current_metrics.reset(token)
        store.save_episode(ep)


async def regenerate_segment(eid: str, seg_id: str, instruction: str | None) -> None:
    """Reviewer hook: rewrite one segment (optionally following a listener instruction),
    re-verify, re-synthesise only that segment, and re-stitch the episode."""
    ep = store.get_episode(eid)
    if ep is None:
        return
    seg = next((x for x in ep.segments if x.id == seg_id), None)
    if seg is None:
        return
    token = current_metrics.set(ep.metrics)
    try:
        async with store.lock(eid):
            ctx = make_context(ep)
            seg.status, seg.error = "writing", None
            label = f' with instruction "{instruction}"' if instruction else ""
            ctx.emit("regenerate", "info", f"Regenerating '{seg.title}'{label}.", seg.id)
            if fast_mode(ep):  # 2 LLM calls: rewrite this part, then fact-check it
                claims = await fast.rewrite_part(ctx, seg, ep.segments, instruction)
                await _record(ctx, seg, fast.new_version(seg, claims, instruction or "regenerate"))
            else:
                current = seg.current.turns if seg.current else None
                seg.candidates = []  # fresh search; memory (reflections) is kept across versions
                budget = 2 if ep.settings.policy in ("adaptive", "uniform") else 1
                await run_search(ctx, seg, ep.segments, budget, policy=ep.settings.policy, accept=get_settings().accept_score,
                                 instruction=instruction, current=current)
                await _finalize_segment(ctx, seg, ep.segments, instruction=instruction or "regenerate")
            await assemble(ctx)
            ctx.emit("regenerate", "info", f"'{seg.title}' is now version {seg.active_version}; episode re-stitched.", seg.id)
    except Exception as e:
        seg.status = "done" if seg.current else "failed"
        seg.error = (getattr(e, "user_message", None) and f"Regeneration failed: {e.service} {e.user_message}.") or "Regeneration failed; the previous version is kept."
        log.error("regen %s/%s failed: %s", eid, seg_id, traceback.format_exc())
    finally:
        current_metrics.reset(token)
        store.save_episode(ep)


async def switch_version(eid: str, seg_id: str, version: int) -> bool:
    ep = store.get_episode(eid)
    if ep is None:
        return False
    seg = next((x for x in ep.segments if x.id == seg_id), None)
    if seg is None or not any(v.version == version for v in seg.versions):
        return False
    async with store.lock(eid):
        seg.active_version = version
        ctx = make_context(ep)
        await assemble(ctx)
        ctx.emit("regenerate", "info", f"'{seg.title}' switched to version {version}.", seg.id)
        store.save_episode(ep)
    return True
