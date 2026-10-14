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

from ..audio.stitch import concat_segments, decode_audio, encode_mp3
from ..clients.base import ServiceError
from ..clients.usage import current_metrics
from ..config import get_settings
from ..ingest import select_chunks
from ..prompts import PROMPT_VERSION
from ..schemas import Episode, Segment, SegmentVersion
from ..store import store
from .allocation import initial_allocation, reallocate
from .claims import verify_and_fix
from .context import AgentContext
from .planner import plan
from .search import best_candidate, run_search
from .voice import synthesize_version

log = logging.getLogger("podcast.orchestrator")


def make_context(ep: Episode) -> AgentContext:
    src = store.get_source(ep.source_id)
    if src is None:
        raise ServiceError("Storage", "lost the uploaded chapter; please upload it again")
    chunks, _ = select_chunks(src, ep.settings.section_ids, get_settings().max_input_words)
    return AgentContext(episode=ep, source=src, chunks=chunks, on_change=lambda: store.save_episode(ep, force=False))


class _Timer:
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
    ver = SegmentVersion(version=(max((v.version for v in seg.versions), default=0) + 1), instruction=instruction,
                         turns=turns, scores=best.scores, claims=claims)
    seg.status = "synthesizing"
    ctx.on_change()
    path = store.episode_dir(ctx.episode.id) / f"{seg.id}_v{ver.version}.wav"
    await synthesize_version(ctx, seg, ver, path)
    seg.versions.append(ver)
    seg.active_version = ver.version
    seg.status = "done"
    seg.error = None


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
    rev = int(time.time())
    name = f"episode_{rev}.mp3"
    (d / name).write_bytes(await asyncio.to_thread(encode_mp3, audio, sr))
    old = ep.audio_file
    ep.audio_file = name
    ep.duration = round(len(audio) / sr, 1)
    if old and old != name and (d / old).exists():
        (d / old).unlink()


async def run_episode(eid: str) -> None:
    ep = store.get_episode(eid)
    if ep is None:
        return
    token = current_metrics.set(ep.metrics)
    s = get_settings()
    try:
        async with store.lock(eid):
            ep.status, ep.error = "running", None
            ctx = make_context(ep)
            ctx.emit("start", "info", f"Agent started (prompts {PROMPT_VERSION}, policy {ep.settings.policy}, "
                     f"{len(ctx.chunks)} passages in scope).")

            # ---- Perceive + Plan ---------------------------------------------------
            if not ep.segments:
                ctx.status("plan", "Reading the chapter and planning the episode", 0.02)
                with _Timer(ep, "plan"):
                    ep.segments = await plan(ctx)
                ctx.emit("plan", "decision", "Outline: " + " | ".join(f"{x.index + 1}. {x.title} (d{x.difficulty})" for x in ep.segments))
                store.save_episode(ep)
            segments = ep.segments
            todo = [x for x in segments if not (x.status == "done" and x.current)]

            # ---- Act + Observe: write with compute-optimal allocation ----------------
            policy = ep.settings.policy
            total_budget = max(len(segments), round(s.budget_per_segment * len(segments)))
            need_write = [x for x in todo if best_candidate(x) is None]
            if need_write:
                with _Timer(ep, "write"):
                    alloc = initial_allocation([x.difficulty for x in need_write], total_budget, policy, s.max_candidates_per_segment)
                    ctx.emit("write", "decision", f"Compute plan ({policy}): total budget {total_budget} writer calls; "
                             "first round " + ", ".join(f"{x.title[:24]}={a}" for x, a in zip(need_write, alloc)))
                    done_count = 0

                    async def first_round(seg: Segment, budget: int) -> int:
                        nonlocal done_count
                        seg.status = "writing"
                        try:
                            spent = await run_search(ctx, seg, segments, budget, policy=policy, accept=s.accept_score)
                        except Exception:  # a bug or outage in one segment must not sink the episode
                            log.error("writing %s failed: %s", seg.id, traceback.format_exc())
                            ctx.emit("write", "error", f"Writing '{seg.title}' failed unexpectedly.", seg.id)
                            spent = budget
                        done_count += 1
                        ctx.status("write", f"Writing: {done_count}/{len(need_write)} segments drafted",
                                   0.15 + 0.45 * done_count / len(need_write))
                        return spent

                    spent = list(await asyncio.gather(*(first_round(x, a) for x, a in zip(need_write, alloc))))
                    for x, sp in zip(need_write, spent):
                        x.budget = sp
                    remaining = total_budget - sum(spent)
                    if policy == "adaptive" and remaining > 0:
                        bests = [(best_candidate(x).scores.overall if best_candidate(x) and best_candidate(x).scores else 0.0) for x in need_write]
                        extra = reallocate(remaining, bests, [x.difficulty for x in need_write], spent, s.accept_score,
                                           s.max_candidates_per_segment)
                        if any(extra):
                            ctx.emit("write", "decision", f"Verifier-guided reallocation of {remaining} leftover call(s): " +
                                     ", ".join(f"{x.title[:24]} +{e} (best {b:.2f})" for x, e, b in zip(need_write, extra, bests) if e))
                            ctx.status("write", "Improving the weakest segments", 0.6)
                            more = await asyncio.gather(*(run_search(ctx, x, segments, e, policy=policy, accept=s.accept_score)
                                                          for x, e in zip(need_write, extra) if e))
                            for x, sp in zip([x for x, e in zip(need_write, extra) if e], more):
                                x.budget += sp
                        else:
                            ctx.emit("write", "decision", f"All segments met the bar; {remaining} call(s) of budget left unused.")
                store.save_episode(ep)

            # ---- Verify claims + synthesise + ASR self-check -----------------------
            ctx.status("voice", "Fact-checking and recording the hosts", 0.62)
            finished = 0
            with _Timer(ep, "voice"):
                async def finish(seg: Segment) -> None:
                    nonlocal finished
                    try:
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
    token = current_metrics.set(ep.metrics)
    s = get_settings()
    seg = next((x for x in ep.segments if x.id == seg_id), None)
    if seg is None:
        return
    try:
        async with store.lock(eid):
            ctx = make_context(ep)
            seg.status, seg.error = "writing", None
            label = f' with instruction "{instruction}"' if instruction else ""
            ctx.emit("regenerate", "info", f"Regenerating '{seg.title}'{label}.", seg.id)
            current = seg.current.turns if seg.current else None
            seg.candidates = []  # fresh search; memory (reflections) is kept across versions
            budget = 2 if ep.settings.policy in ("adaptive", "uniform") else 1
            await run_search(ctx, seg, ep.segments, budget, policy=ep.settings.policy, accept=s.accept_score,
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
