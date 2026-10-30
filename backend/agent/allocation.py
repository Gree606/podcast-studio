"""Compute-optimal test-time allocation, after Snell et al. (2024).

Two ideas from the paper, adapted from "questions" to "podcast segments":

1. Allocate test-time compute by difficulty. Instead of the same N for every
   segment, a fixed episode budget is split so harder segments get more writer
   calls. Difficulty is first *predicted* (planner estimate + text prior), then
   *corrected* by the verifier: after the first round, leftover budget goes to
   segments whose best verifier score is still below the acceptance bar.
2. Easy problems benefit most from sequential revision, hard ones from more
   parallel samples. ``search_shape`` splits a segment's budget accordingly.

All functions here are pure, so they are unit-tested directly.
"""
from __future__ import annotations

import math


def _largest_remainder(weights: list[float], total: int, caps: list[int]) -> list[int]:
    """Split ``total`` in proportion to ``weights`` (largest-remainder rounding), never exceeding ``caps``;
    whatever a capped entry cannot take is redistributed among the others."""
    out = [0] * len(weights)
    remaining = total
    active = [i for i in range(len(weights)) if caps[i] > 0]
    while remaining > 0 and active:
        wsum = sum(weights[i] for i in active) or float(len(active))  # all-zero weights: remainders go round-robin
        raw = {i: remaining * weights[i] / wsum for i in active}
        floors = {i: min(int(math.floor(raw[i])), caps[i] - out[i]) for i in active}
        given = sum(floors.values())
        for i, f in floors.items():
            out[i] += f
        remaining -= given
        if remaining > 0:
            order = sorted(active, key=lambda i: (raw[i] - math.floor(raw[i]), weights[i]), reverse=True)
            for i in order:
                if remaining == 0:
                    break
                if out[i] < caps[i]:
                    out[i] += 1
                    remaining -= 1
        active = [i for i in active if out[i] < caps[i]]
    return out


def initial_allocation(difficulties: list[int], total_budget: int, policy: str, max_per: int) -> list[int]:
    """Writer calls per segment for the first round.

    single / no_critic : 1 each (the N=1 baselines)
    uniform           : total budget split evenly (classic best-of-N)
    adaptive          : 1 each up front; the rest is held back for verifier-guided
                        reallocation, except hard segments (difficulty >= 4) get a
                        second parallel sample immediately.
    """
    n = len(difficulties)
    if n == 0:
        return []
    if policy in ("single", "no_critic"):
        return [1] * n
    total_budget = max(total_budget, n)
    if policy == "uniform":
        base, extra = divmod(total_budget, n)
        return [min(max_per, base + (1 if i < extra else 0)) for i in range(n)]
    alloc = [1] * n
    spare = total_budget - n
    for i in sorted(range(n), key=lambda i: -difficulties[i]):
        if spare <= 0:
            break
        if difficulties[i] >= 4 and alloc[i] < max_per:
            alloc[i] += 1
            spare -= 1
    return alloc


def reallocate(remaining: int, best_scores: list[float], difficulties: list[int], spent: list[int],
               accept: float, max_per: int) -> list[int]:
    """Second round (adaptive only): give leftover budget to segments still below ``accept``,
    weighted by how far below they are and by predicted difficulty."""
    weights = []
    caps = []
    for s, d, used in zip(best_scores, difficulties, spent):
        gap = max(0.0, accept - s)
        weights.append(gap * (0.5 + d / 5))
        caps.append(max(0, max_per - used) if gap > 0 else 0)
    if remaining <= 0 or not any(c > 0 for c in caps):
        return [0] * len(best_scores)
    return _largest_remainder(weights, remaining, caps)


def search_shape(budget: int, difficulty: int, policy: str) -> tuple[int, int]:
    """Split a segment's budget into (parallel samples, sequential revisions).

    Snell et al.: easier prompts gain most from sequential revisions of one draft;
    harder prompts need broader parallel exploration. Uniform best-of-N is all-parallel.
    """
    if budget <= 1:
        return 1, 0
    if policy == "uniform":
        return budget, 0
    if difficulty <= 2:
        return 1, budget - 1
    if difficulty == 3:
        par = math.ceil(budget / 2)
        return par, budget - par
    return budget - 1, 1
