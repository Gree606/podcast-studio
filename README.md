# Podcast Generation Agent Project

Podcast Studio: Project Context Brief

Purpose of this file: give a coding/planning agent everything it needs to build, harden and defend this project. It covers the course context, the idea, why we chose it, the current code state, and the implementation plan that maximizes marks. Read it fully before changing anything.

Prepared: 2026-10-14. Project due: 2026-10-31 (17 days).

1. Context
Course: Recent Advances in Generative AI, DSAI, IIT Madras (Jul to Nov 2026), instructor Mitesh M. Khapra. Theme: "from inference-time compute to agentic systems".
Student: Gree. Goal: top grade (S).
Grading: Paper-reading cheat sheets 10%, Poster 1 20% (Sep 21-22, done), Poster 2 20% (Nov 2-3), Project 30%, Perusall 20%.
Project (30%) splits into:
Code, 10%: implementation quality of the agentic system.
In-the-Wild Evaluation, 20%: TAs open the live link as real users and grade on features, ease of use, output quality.
Deliverable: a working live link + code, due Oct 31.
AI policy: unrestricted AI tools for coding on the project. Use them heavily.
Course definition of an agent (slide 3): "A system that uses an LLM to decide actions, call tools, and iterate toward a goal", with a Perceive, Plan, Act, Observe loop. The system must visibly show that loop.
Papers read so far (what the tutor expects us to connect to):
Snell et al., Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters: adaptive, difficulty-aware test-time compute (best-of-N with a verifier, compute-optimal allocation) can match or beat much larger models.
Yao et al., ReAct: interleave reasoning traces with actions (tool calls); improves grounding and interpretability.
Shinn et al., Reflexion: verbal self-feedback stored in an episodic memory improves later attempts without weight updates.
Liu et al., ToolACE: accurate, diverse tool-calling data; dual-layer (rule-based + model-based) verification.
Qian et al., ToolRL: fine-grained, principled reward design for tool use (GRPO); coarse answer-matching rewards are insufficient.
Project options offered by the course

Dubbing Studio, Translation Studio, Auto-Assessment, Podcast Generation, Video Generation, Infinite OCR (or own idea, subject to approval). Each has an Input / Output / Reviewer / Bar spec. For Podcast Generation:

Field	Spec
Input	An NCERT textbook chapter or section
Output	An engaging, conversational podcast-style audio explainer
Reviewer	Request regeneration of specific segments
Bar	Sounds like a real podcast, not text-to-speech; faithful to the source
2. The idea we chose: Podcast Studio

One line: upload an NCERT chapter, get a two-host conversational podcast that is faithful to the chapter, with a visible agent reasoning trace and per-segment regeneration.

Why Podcast Generation (decision rationale)
Best fit to the course thesis. The course arc is test-time compute, then agent capabilities. Podcast generation is naturally a multi-stage agent: plan, write, critique, revise, synthesize, verify. Every stage maps to a paper the tutor assigned, so the project reads as applying the course rather than merely using an LLM.
Room for a real agentic loop. Translation and Dubbing are largely fixed pipelines (extract, translate, render). Infinite OCR is mostly perception. Podcast has genuine decision points (what to cover, whether a draft is good enough, whether the audio matches the script), which is what "observe-plan-act" needs.
Fits the grading rubric. In-the-wild grading rewards features, ease of use and output quality. Podcast output is audible and immediately judgeable by a TA in a few minutes, and the reviewer feature (regenerate one segment) is a concrete, demoable feature.
Matches the API access we have. We have an LLM (Gemma), TTS, ASR and OCR via API. Podcast uses all four (LLM for reasoning, TTS for voice, ASR for self-check, OCR for scanned chapters). No GPU hosting is needed.
Defensible technical story. "A small model plus adaptive inference-time compute" is exactly Snell et al.'s finding, and we can measure it in an ablation.
Risk is manageable. Video generation is too heavy for the timeline. Podcast is audio-only, so one pipeline and one deployable service.

MT (machine translation) is not required. Optional stretch: multilingual podcast (translate then script, or script directly in the target language).

3. Architecture (how each stage earns marks)
PDF -> [1] Ingestion (text extract, OCR fallback)
    -> [2] Planner (ReAct-style: reason + retrieve/define tools, verify facts) -> outline of segments w/ difficulty
    -> [3] Writer x Critic loop per segment (Reflexion: verbal feedback + episodic memory)
    -> [4] Compute-optimal best-of-N (more candidates for harder segments; critic = verifier)
    -> [5] TTS per turn, distinct host voices, stitched
    -> [6] ASR self-check (transcribe own audio, WER vs script, auto re-synthesize)
    -> [7] Segment-addressable regeneration (reviewer hook)
Stage	Paper link	What the tutor should see
Planner with retrieve/define tools and grounding verification	ReAct, ToolACE	Visible Thought/Action/Observation trace in the UI; rule-based + model-based tool-result verification
Writer/Critic with verbal feedback and memory	Reflexion	Critic feedback shown per segment; memory buffer across retries
Difficulty-based sample allocation, critic as verifier	Snell et al.	Adaptive N per segment; ablation vs uniform N and N=1
Multi-axis rubric (faithfulness, engagement, tone, pacing)	ToolRL (reward design principle)	Fine-grained scores, not pass/fail
ASR self-check loop	Course "agent" definition	Agent observes its own output and corrects it

Honesty note for the poster Q&A: we are not training with RL. ToolRL informs the rubric/reward design only. Do not claim otherwise.

4. Current code state (what exists, what is weak)

Location of the scaffold: podcast-studio/ (zip delivered earlier). FastAPI backend + single-file frontend served from the same app.

Done and verified with mocked APIs (scripts/smoke_test.py passes): schemas, config, four API clients (templates), ingestion, tools, planner, writer, critic, compute-optimal allocation, audio stitch + WER, orchestrator, API routes, frontend, README. The smoke test caught and fixed one real bug (segment audio not persisted, which broke single-segment regeneration).

Not yet done / known weaknesses (the agent should fix these, in this priority order):

Never run against the real APIs. Client request/response shapes are guesses. First task: wire the real Gemma/TTS/ASR/OCR endpoints and test on a real chapter.
POST /generate is a blocking call. A full episode takes minutes; hosted platforms and browsers time out. Convert to a background job with progress polling or SSE (status per stage/segment). This is the single biggest risk for the in-the-wild score.
State is in-memory. Restarts lose episodes; multiple workers break it. Persist to SQLite/disk, keep a single worker, or both.
OCR fallback is a stub (_render_page_to_png raises NotImplementedError). Use PyMuPDF.
WAV assumption. Stitching assumes WAV from TTS; transcode MP3 to WAV (pydub/ffmpeg) or switch stitching.
The "ReAct" planner is only partly interleaved. It does one narrated scratchpad, then one outline call, then post-hoc retrieve/verify. define is never invoked. Upgrade to a real tool-calling loop (model emits action, we execute, feed back observation, repeat until done) so the ReAct claim holds under questioning.
Critic and writer are the same model. Risk of self-preference bias and score inflation. Mitigate: different temperature/prompt, a stricter rubric with concrete anchors, a deterministic check layer (coverage of must-mention facts, length, banned phrasing), and optionally a pairwise comparison for selection.
Faithfulness is only checked at planning time. Add a post-write claim check: extract factual claims from the final script and verify each against retrieved chapter text. Surface unsupported claims.
Segment audio is stitched with equal silence only. Naturalness needs more (see section 5).
No tests beyond the smoke test, no logging, no Dockerfile, no cost/latency tracking.
5. Implementation plan to maximize marks

Rule of thumb from the rubric: 20% of the project grade is a TA using the live link cold. A smaller feature set that works smoothly beats a larger one that breaks. Prioritize reliability, then output quality, then extra features.

P0: must work for any TA, first try (do these first)
Real APIs wired and tested on at least 5 NCERT chapters (mix of subjects and classes; digital and scanned PDFs).
Async job + live progress UI ("Planning, Writing 3/5, Synthesizing, Self-checking"). Show the reasoning trace as it happens.
Zero-friction start: a "Try a sample chapter" button (pre-bundled chapter) next to upload, so a TA can get a result in one click. Also accept pasted text and a chapter/section selector.
Robust failure handling: every external call has timeout, retry with backoff, and a human-readable error. If one segment fails, the episode still completes and flags that segment. Never show a stack trace.
Deployed early (target: by Oct 20) on Render/Railway/Fly with env vars, health check, and a keep-warm so there is no cold-start surprise. Test the live URL from a different network/device.
Concurrency safety: several TAs may test at once. Per-episode isolation, per-job locks, rate-limit awareness (queue LLM calls with a concurrency cap).
Length control: target 5-10 minutes per episode, cap input size, tell the user the expected wait.
P1: output quality (the "sounds like a real podcast" bar)
Script naturalness: two distinct personas (curious host + explainer), contractions, short reactions ("right", "wait, so..."), varied turn length, analogies and Indian-context examples, a hook intro, signposting between segments, and a recap. Forbid lecture-style monologues over N words. Enforce with the critic rubric and a deterministic length check.
Audio naturalness: per-turn voice with prosody (SSML or style tags if the TTS supports them), natural pauses (longer between topics than between turns), light crossfades, loudness normalization across segments, optional short intro/outro music sting. Avoid reading numbers/symbols/units literally: add a text-normalization step (e.g. "H2O" to "H two O", "5 m/s" to "five metres per second").
Faithfulness: claim-level post-check (section 4, item 8); show a "Sources" panel mapping each segment to the chapter passages used; instruct hosts not to introduce outside facts; flag and rewrite unsupported claims.
Pedagogy: a one-line "what you'll learn" opener and a 3-point recap; keep NCERT terminology exact; skip or briefly reference exercises and activities rather than reading them aloud.
P2: features TAs will test (reviewer + ease of use)
Segment regeneration with an optional natural-language instruction ("shorter", "add an example", "more casual"). Only that segment re-runs and only its audio re-synthesizes; the full episode re-stitches seamlessly. Keep version history with undo per segment, and let the user play old vs new.
Transcript view synced with audio (click a line to jump).
Download the mp3 and the script.
Controls: length (short/medium/long), tone (casual/serious), language if supported.
Clear UI states: progress, errors, empty states, mobile-friendly layout, accessible labels.
P3: evaluation harness (what makes the code grade and the poster Q&A strong)

Build a small eval/ script that runs the pipeline over a fixed set of 5-8 chapters and reports:

Faithfulness: claim-support rate (LLM judge or entailment against source passages).
Naturalness: rubric score (LLM judge) plus a small human check (ask 3-5 classmates to rate 1-5 in a form).
Audio integrity: ASR WER per segment, before/after self-check.
Cost and latency: tokens, calls, seconds per episode.
Ablations (the Snell et al. story): (a) no critic, (b) critic with N=1, (c) uniform N, (d) adaptive N at the same total budget. Show quality vs compute. Even a small table is powerful evidence.
P4: code quality (the 10% code grade)
Clean modules (already structured), type hints, docstrings that explain why.
pytest unit tests for: allocation, WER, stitching, schema validation, JSON repair, retrieval; keep smoke_test.py as an integration test.
Structured logging with a per-episode ID; log tokens/latency per stage.
Pinned requirements.txt, .env.example, Dockerfile, a clear README (setup, architecture diagram, how to run evals, how to deploy), no secrets in git.
Prompts in separate files/constants, versioned, so tuning is traceable.
Suggested timeline (today is Oct 14; deadline Oct 31)
Dates	Goal
Oct 14-16	Wire real APIs; first real end-to-end episode; fix OCR rasterizer; WAV/mp3 handling
Oct 17-19	Async jobs + progress UI; persistence; error handling; real tool-calling planner
Oct 20	First deployment to the live URL; cold test from another device
Oct 21-24	Quality pass: prompts, naturalness, text normalization, post-write claim check, audio polish
Oct 25-27	Eval harness + ablation table; segment history/undo; sample-chapter button; UI polish
Oct 28-29	Cold-test with classmates on unseen chapters; fix what breaks; tests + README + Dockerfile
Oct 30	Code freeze; final deploy; verify link; record a backup demo video
Oct 31	Submit link + code; do not make risky changes

Poster 2 is Nov 2-3: reuse the ablation table, trace screenshots and architecture diagram.

6. Defending it (poster Q&A and TA questions)

Likely questions and the honest answers:

"Why not just use a bigger model?" Show the ablation: adaptive best-of-N with a small model vs single-shot; cite Snell et al. Be clear about the budget and where it stops helping.
"Is this really ReAct?" After upgrading the planner to a true tool loop, yes. Show the trace.
"Did you do RL?" No. The rubric is reward-design inspired by ToolRL; selection is best-of-N with a verifier, not policy training.
"Can the critic be trusted?" Acknowledge self-preference bias; point to deterministic checks, the claim-level verifier, and human ratings in the eval.
"What fails?" Know the failure cases (dense math/derivations, tables/figures, scanned regional-language pages, number/symbol pronunciation) and what the system does about each.
7. Operating rules for the agent
Never commit API keys; read from env only.
Keep provider-specific code inside backend/clients/.
Prefer the smallest change that fixes a problem; re-run scripts/smoke_test.py after backend changes, and add a test for each bug fixed.
Do not expand scope (no video, no extra languages) until P0 and P1 are done and deployed.
Any claim in docs or poster material must match what the code actually does.
When unsure about an API's real behavior, test it with a tiny call and record the finding in the README.
8. Open items the agent must ask the user about
Exact base URLs, auth, and request/response shape for the Gemma, TTS, ASR and OCR APIs (sample curl and response for each), and whether TTS returns WAV or MP3.
Which languages the NCERT chapters will be in (English only, or Hindi/regional), and which TTS voices/languages are available.
Credit limits and rate limits per API (affects the sample budget default; the working budget estimate is about 60-80k tokens per episode, so $20-30 LLM credits is a comfortable ask, roughly $50-65 across all four services).
Hosting choice and any course-provided deployment resources.
Whether the course requires specific submission artifacts (README, report, video) beyond "working link + code".
