# Podcast Studio

Turn an NCERT chapter into a two-host conversational podcast. The podcast stays faithful to the book, the agent's reasoning trace is visible, and you can regenerate any single segment.

*Course project for Recent Advances in Generative AI (IIT Madras, 2026). Project brief: [`docs/PROJECT_BRIEF.md`](docs/PROJECT_BRIEF.md).*

## What it does

1. **Input.** Upload an NCERT chapter PDF (scanned pages go through OCR), paste text, or pick a bundled sample chapter. Then choose which sections to cover, the length and the tone.
2. **Agent.** A small, fast LLM (Gemini Flash; any OpenAI-compatible model also works) runs a Perceive, Plan, Act, Observe loop. The UI shows it live.
3. **Output.** An MP3 episode with a transcript that follows the audio (click a line to jump to it), a per-segment "Sources" panel and a claim check.
4. **Listen & play.** A kid-friendly player ("Chapter Chat", ages 10-15): the two cartoon hosts light up as they speak, the line being spoken is highlighted, and one tap ("🐣 Simpler", "🍎 Add an example", "🎈 Make it fun") rewrites just that part. A quiz at the end is built only from verified key facts (fill-in-the-blank; a wrong option is never a word from the cited passage). A **🎓 Teacher view** shows the agent trace, quality & compute, claim check, sources and every version. The previous UI is still at `/classic`.
5. **Reviewer.** Regenerate any one segment, optionally with an instruction ("simpler", "add an example"). Only that segment is rewritten and re-recorded, and the episode is re-stitched. Every version is kept, and you can compare versions or switch back.

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
cp .env.example .env                 # optional: non-secret settings (models, voices, knobs)
python scripts/init_secrets.py       # creates ~/.config/podcast-studio/secrets.env (owner-only)
#   ...open that file and replace the three DUMMY_ values: LLM_API_KEY, TTS_API_KEY, ASR_API_KEY
python scripts/check_apis.py         # one tiny call per API: confirms keys, voices, ASR script
scripts/run_local.sh start           # background server on :8000 (logs: data/server.log)
scripts/run_local.sh status | stop | restart
```

**Where the keys live.** API keys are kept *outside* the repo, in `~/.config/podcast-studio/secrets.env` (or the file named by `PODCAST_SECRETS_FILE`), so nothing working inside the repo ever sees them. Load order, later wins: repo `.env`, then the secrets file, then real environment variables (so a host like Render can still set keys as env vars). The app and `/api/config` only ever report *whether* each key is set, never its value.

Each service switches from its offline fake to the real API as soon as its own key is set (so real voices work even while the LLM key is a placeholder). `OCR_MODE=bodhan` reads scanned pages with Bodhan Indic-OCR (`OCR_API_KEY`). The ASR self-check sends the language (default: the TTS language) and paces itself to 14 calls/minute, Bodhan's measured sustained limit.

While any required key is still a `DUMMY_` placeholder, the app runs in **demo mode** (placeholder scripts and beeps) and says so in a banner. On shared HPC login nodes, run `source scripts/dev_env.sh` first (it caps BLAS threads).

## Providers

| Service | Used for | Wire format (see `backend/clients/`) |
|---|---|---|
| **Gemini** (`gemini-flash-latest`) | planner, writer, critic, claim checks, OCR of scanned pages | `POST /v1beta/models/{model}:generateContent`, header `X-goog-api-key`. System prompt goes in `systemInstruction`; images as `inline_data`; `responseMimeType: application/json` for structured calls; thought parts are skipped and thinking tokens are counted. `LLM_PROVIDER=openai` switches to any OpenAI-compatible server (e.g. vLLM + Gemma). |
| **Sarvam** `bulbul:v4-flash` | the two voices | `POST https://api.sarvam.ai/text-to-speech`, header `api-subscription-key`, body `{text, language_code: "en-IN", speaker, model, pace, speech_sample_rate}`, returns `{"audios": [base64 WAV]}`. Note: the field is `language_code`, not `target_language_code`, and the SDK's `convert_stream` yields chunks rather than bytes (checked in `sarvamai` 0.1.37). Default voices `simran_en_conversation` (host) and `dev_en_conversational` (explainer). Turns over 900 characters are split at sentence boundaries. |
| **Bodhan** `indic-transcribe` | audio self-check | `POST https://api.bodhan.ai/v1/audio/transcriptions`, multipart `file` + `model`, header `Authorization: <key>` (no `Bearer`). Optional: if not configured, the self-check is skipped. If it answers in a different script (e.g. Devanagari for English speech), the self-check stands down instead of re-recording everything. |

All calls have timeouts and retries with backoff. 429 responses wait for the server's `Retry-After` or Google `retryDelay`. Bad keys produce a plain-language error in the UI.

## Tests

```bash
pytest -q                                  # 54 unit tests
MOCK_MODE=1 python scripts/smoke_test.py   # end to end with in-process mocks
python scripts/contract_test.py            # REAL client code vs local fakes of Gemini/Sarvam/Bodhan
                                           # (wire formats, auth headers, 429 retry, bad keys, regeneration)
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

Every provider URL comes from environment variables (see `.env.example`); keys come from the secrets file or environment variables. No secrets live in git.

## Known limitations

- Dense derivations and tables are summarised rather than read out.
- Figures are not described.
- English (Indian English voices) only for now. Sarvam and Bodhan both support Indian languages, which makes a Hindi/regional episode a natural extension.
- Number and symbol reading is rule-based (`backend/textnorm.py`), so rare notation may still be read literally.
