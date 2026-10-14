# Podcast Studio

Turn an NCERT chapter into a two-host conversational podcast. The podcast stays faithful to the book, the agent's reasoning trace is visible, and you can regenerate any single segment.

*Course project for Recent Advances in Generative AI (IIT Madras, 2026). Project brief: [`docs/PROJECT_BRIEF.md`](docs/PROJECT_BRIEF.md).*

## What it does

1. **Input.** Upload an NCERT chapter PDF (scanned pages go through OCR), paste text, or pick a bundled sample chapter. Then choose which sections to cover, the length and the tone.
2. **Agent.** A small open model (Gemma) runs a Perceive, Plan, Act, Observe loop. The UI shows it live.
3. **Output.** An MP3 episode with a transcript that follows the audio (click a line to jump to it), a per-segment "Sources" panel and a claim check.
4. **Reviewer.** Regenerate any one segment, optionally with an instruction ("simpler", "add an example"). Only that segment is rewritten and re-recorded, and the episode is re-stitched. Every version is kept, and you can compare versions or switch back.

## Architecture

```
PDF/text ─► ingest.py  (PyMuPDF text layer; vision-LLM OCR for scanned pages;
                        NCERT clean-up: faux-bold duplicates, drop caps, running heads,
                        exercises skipped) ─► sections ─► ~170-word passages (BM25 index)
        │
        ▼
 [Plan]   agent/planner.py   ReAct loop. Each turn the model emits {thought, action, input}.
                             Tools: list_sections / search / read / define / finish.
                             Key facts cite passages, then pass a dual-layer check
                             (agent/tools.py: rule-based lexical support + LLM judge).
        │  outline: segments + key facts + difficulty (LLM estimate blended with a text prior)
        ▼
 [Act]    agent/orchestrator.write_segments + agent/allocation.py
                             Fixed episode budget of writer calls. Round 1 is sized by
                             predicted difficulty; leftover budget goes to segments the
                             verifier still scores below the bar. Easy segments get
                             sequential revisions; hard ones get parallel drafts.
          agent/search.py    draft ─► critic ─► reflection into episodic memory ─► revise
          agent/critic.py    deterministic checks (monologues, length, key-fact coverage,
                             banned phrasing) + 5-axis anchored rubric; overall is capped
                             at faithfulness + 1
        │
        ▼
 [Observe] agent/claims.py   extract every factual claim ─► BM25 evidence ─► judge;
                             one targeted rewrite; anything still unsupported is shown in the UI
          agent/voice.py     text normalisation (H2O, 5 m/s, years, symbols) ─► TTS per turn
                             ─► ASR transcribes its own audio ─► WER > 25% ⇒ re-synthesise
        │
        ▼
 audio/stitch.py             decode WAV/MP3, trim, loudness-match, pauses that vary with the
                             conversation, intro/outro chime, MP3 export, per-turn timestamps
```

| Stage | Paper | What you can see in the app |
|---|---|---|
| Tool-using planner | ReAct (Yao et al.) | "Agent trace" tab: Thought, Action, Observation |
| Fact verification of tool results | ToolACE (Liu et al.): dual-layer verification | "Verified N key facts: x failed rule check, y failed model check" |
| Draft, critique, lesson in memory, revise | Reflexion (Shinn et al.) | Critic feedback and "Reflection stored in memory" per segment |
| Difficulty-aware budget, verifier-guided reallocation | Snell et al. 2024: compute-optimal test-time scaling | "Quality & compute" tab: difficulty, writer calls, every draft's score |
| Fine-grained multi-axis reward | ToolRL (Qian et al.): reward design only | Five rubric axes plus a rule penalty |
| ASR self-check of its own audio | Course definition of an agent (observe and correct) | WER per turn, re-recorded lines |

**Honesty notes:**
- No RL training happens here. ToolRL inspired the rubric design, and selection is best-of-N with a verifier.
- The critic and the writer are the same base model. The deterministic checks, the claim checker and the separate eval judge are there to limit self-preference.

## Run locally

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env              # fill in endpoints; leave LLM_BASE_URL empty for demo mode
uvicorn backend.main:app --port 8000   # single worker on purpose
open http://localhost:8000
```

If no LLM endpoint is set, the app runs in **demo (mock) mode**, with placeholder scripts and voices, and shows a banner saying so.

On shared HPC login nodes, run `source scripts/dev_env.sh` first (it caps BLAS threads).

## Tests

```bash
pytest -q                                  # 42 unit tests: allocation, WER, text norm, JSON repair,
                                           # BM25, audio stitching, ingestion, tools, planner loop, critic rules
MOCK_MODE=1 python scripts/smoke_test.py [chapter.pdf]   # end to end: upload → episode → regenerate → switch version
```

## Evaluation and ablation

```bash
python eval/run_eval.py --chapters chem-reactions,crop-production,nationalism-europe --length short [--audio]
# optional independent judge model:
EVAL_JUDGE_BASE_URL=... EVAL_JUDGE_MODEL=... python eval/run_eval.py
```

The planner runs once per chapter. Four policies then write from the same outline:
- `no_critic`: N = 1, no verifier.
- `single`: N = 1, the critic only scores.
- `uniform`: best-of-N with the budget split evenly.
- `adaptive`: our policy.

The harness reports:
- writer calls, tokens and time;
- independent judge scores (faithfulness, naturalness, engagement, pedagogy);
- claim-support rate;
- with `--audio`, ASR WER on first takes vs. after the self-check.

Output goes to `eval/results/<timestamp>/results.md`.

## Deploy

The app is a single Docker image (`Dockerfile`), and `render.yaml` is a Render blueprint for it. Deployment needs:
- **one worker** and a **persistent disk** at `DATA_DIR`. Job state and locks are in-process, and episodes are JSON plus audio on disk. Restarts mark running jobs as "interrupted", and the user can press Resume.
- an always-on plan, so there is no cold start while a TA is waiting;
- `/api/health` as the health check.

Every provider URL and key comes from environment variables (see `.env.example`). No secrets live in git.

## API endpoint notes (findings so far)

| Service | Interface the app expects | Status (2026-10-15) |
|---|---|---|
| LLM | OpenAI `/v1/chat/completions` (vLLM serving Gemma 4). The tool loop uses prompted JSON because the servers are not launched with `--enable-auto-tool-choice`. Gemma's empty `<\|channel>thought…<channel\|>` block is stripped. The 31B server's context is 20,480 tokens, so planner observations are truncated. | Cluster deployments currently down. A public endpoint is needed. |
| TTS | `TTS_API_STYLE=openai` (`/v1/audio/speech`) or `json` (`/synthesize`, base64 or raw audio). Returns WAV or MP3; both are decoded. | No HTTP TTS service exists yet. The in-house Robin model is offline/batch only. |
| ASR | OpenAI-style `/v1/audio/transcriptions` (multipart `file`, `language`). This is the Canary NIM HTTP shape. Riva gRPC would need an adapter. | Down. The self-check is skipped automatically if ASR is not configured. |
| OCR | Gemma vision via `image_url` data URI, image before text. | Same as LLM. |

## Known limitations

- Dense derivations and tables are summarised rather than read out.
- Figures are not described.
- English only for now. Both the TTS and ASR stacks support Indian languages, which makes this a natural extension.
- Number and symbol reading is rule-based (`backend/textnorm.py`), so rare notation may still be read literally.
