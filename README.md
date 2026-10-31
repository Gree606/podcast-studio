# Mission 606: Earth Unknown

Turn an NCERT chapter into a two-host podcast. **Gree**, an astronaut from India, explains the chapter to **Error606**, a curious alien visiting Earth. The episode stays faithful to the book, every spoken line is fact-checked against the chapter, and you can regenerate any single part.

*Course project (Podcast Generation) for Recent Advances in Generative AI, IIT Madras, Jul–Nov 2026.*

| Spec | What this app does |
|---|---|
| **Input:** an NCERT chapter or section | A PDF (scanned pages go through OCR), up to 6 phone photos of pages, pasted text, or one of 9 bundled NCERT sample chapters (5 English, 2 Hindi, 2 Malayalam). Pick the sections to cover. |
| **Output:** a conversational podcast | An MP3 of about 2 minutes with a synced transcript, in any of 12 languages: English, Hindi, Hinglish, Bengali, Tamil, Telugu, Kannada, Malayalam, Marathi, Gujarati, Punjabi and Odia. |
| **Reviewer:** regenerate specific segments | Regenerate one part, optionally with an instruction such as "simpler" or "add an example". Only that part is rewritten, re-recorded and re-stitched. Every version is kept. |
| **Bar:** sounds real and stays faithful | Two distinct conversational voices, natural reactions and pauses. The fact-check badge counts the lines the chapter supports. |

## Using it

1. Open the link and give Error606 the secret code. The code is required because generating an episode spends API calls.
2. Pick a sample chapter, upload a PDF, take photos or paste text, then choose the language. **☰ Menu → ⚙️ Settings** (also reachable from the launch screen's "Studio" line) picks, for each new episode:
   - **Mode:** ⚡ Quick (default, 2 LLM calls) or 🧠 Deep agent (plans, drafts and critiques).
   - **Script AI:** Gemini with its own key, or the same Gemini models through OpenRouter.
   - **Voices:** Sarvam bulbul or Bodhan indic-speak.

   Each episode remembers its choices, so regenerating a part uses the same AI and voices. Options whose key isn't set on the server are greyed out (`backend/providers.py`).
3. Start listening as soon as the first part is recorded; the remaining parts are recorded in the background. The dock at the bottom opens:
   - **Script:** the transcript, highlighted line by line as it plays. Click a line to jump to it.
   - **Ask Gree:** type a doubt in any language. Gree answers in 2–5 spoken sentences using only the chapter, and cites the section and page. If the chapter doesn't cover the question, she says so; off-topic questions are steered back to the chapter. Tap 🔊 to hear the answer. (`backend/ask.py`)
   - **Quiz:** questions built only from verified facts. A wrong answer has an "Ask Gree why" button.
   - **Agent:** what actually ran for this episode, including the fact-check, sources, compute and (in Deep mode) the full agent trace. Steps that didn't run are not shown.
   - **Save:** download the MP3 and the transcript.
4. To change something, open a part and regenerate it. Old and new versions can be compared and switched.

**Privacy and ownership.** There is no public list of episodes; each episode is reachable only through its link. The browser that made an episode holds an owner cookie, and only that browser can regenerate parts, switch versions or restart. Anyone with the link can listen, take the quiz and ask Gree. `/api/config` reveals no server paths or key values. Expensive calls are rate-limited per client, and repeated wrong codes are slowed down.

**Never a frozen screen.** Each LLM call times out after 45 s and falls back to the next model in the list. The progress bar keeps moving during a long call. After 100 s (10 min for Deep) the page says "Taking longer than usual" and shows a Try again button (`POST /api/episodes/{id}/restart`). Restarting the server marks running jobs as interrupted, and they can be resumed.

**Layout.** The app works on desktop (two columns), on phones down to 320 px wide, and on rotated phones. It has pixel-art sprites, an animated space scene and optional sound effects (there is a sound switch on every page).

## How it works

Both modes share the same ingestion, tools and audio stages. The mode only changes how the script is written.

```
PDF / photos / text ─► ingest.py   PyMuPDF text layer; OCR (Bodhan Indic-OCR, or the LLM's vision input)
                                   for scanned pages and photos; NCERT clean-up (faux-bold duplicates,
                                   drop caps, running heads, exercises skipped)
                     ─► sections ─► ~170-word passages ─► BM25 index (retrieval.py)

⚡ QUICK  (agent/fast.py, default)                 🧠 DEEP AGENT  (agent/orchestrator.py)
  Perceive  retrieve the passages in scope           Plan     ReAct planner (planner.py): {thought, action, input}
  Act       call 1: the whole episode as 3 parts,             with list_sections / search / read / define / finish.
            each citing the passages it used                  Key facts pass a rule-based and an LLM check (tools.py)
  Observe   call 2: check EVERY spoken line against  Act      difficulty-aware writer budget (allocation.py);
            those passages; rewrite only the                  draft ─► critic ─► reflection ─► revise (search.py)
            lines the passages don't support         Observe  claim extraction ─► BM25 evidence ─► judge (claims.py)

                     ─► agent/voice.py  text normalisation (H2O, 5 m/s, years, symbols) ─► TTS per line
                                        ─► ASR transcribes its own audio ─► WER > 25% ⇒ re-record
                     ─► audio/stitch.py trim, loudness-match, conversational pauses, intro/outro, MP3 + timestamps
```

**Why Quick is the default.** The Deep agent makes 25–40 LLM calls per episode. That is too many for free-tier quotas, and for a 2-minute script it added little over a single call to a thinking model. The steps a single call cannot do stay as tool steps: checking the script against the book, listening back to its own audio, OCR and per-part regeneration. The Deep agent is still available per episode, so the two can be compared, and the eval harness measures exactly that trade-off.

| Idea from the course | Where it is used |
|---|---|
| **Snell et al. 2024**, compute-optimal test-time scaling | Deep: difficulty-aware writer budget, with leftover calls going to segments the verifier still scores low. Quick vs Deep is itself a compute-allocation choice, and `eval/` ablates it. |
| **ReAct** (Yao et al.) | Deep planner: an interleaved Thought / Action / Observation loop over chapter tools, visible in the Agent panel. |
| **Reflexion** (Shinn et al.) | Deep writer: the critic's feedback becomes a stored reflection that the next draft uses. |
| **ToolACE** (Liu et al.), dual-layer verification | Key facts must pass both a rule-based lexical check and an LLM check. |
| **ToolRL** (Qian et al.), fine-grained reward design | A 5-axis anchored rubric plus deterministic penalties. Overall score is capped at faithfulness + 1. |
| Course definition of an agent: perceive → plan → act → observe | Both modes observe their own output: the fact-check reads the script, and ASR listens to the audio. |

**Honesty notes.**
- No RL training happens here. ToolRL inspired the rubric only, and selection is best-of-N with a verifier.
- The writer and the checker are the same base model. The deterministic checks, per-line evidence and a separate eval judge are there to limit self-preference.
- The fact-check badge counts spoken lines (Quick) or extracted claims (Deep), never planner notes. It looks like "14/16 lines supported by your chapter · 1 corrected · 1 flagged".

**Teaching rules.** The writer must explain a method with one worked example, call an unbalanced chemical equation *skeletal*, and keep cause and effect the right way round. The checker flags lines that break these rules.

## Code map

```
backend/
  main.py          FastAPI routes, access gate, owner cookie, rate limits, static frontend
  jobs.py          background episode jobs, progress events, resume / restart
  ingest.py        PDF / text ─► sections;  photos.py  phone photos ─► OCR'd pages
  retrieval.py     BM25 over passages (also used by Ask Gree and the fact-check)
  agent/           fast.py (Quick) · planner, writer, critic, search, allocation, claims (Deep) · voice.py
  audio/           stitch.py, wer.py
  clients/         llm.py (Gemini / OpenRouter / OpenAI-compatible), tts.py, asr.py, ocr.py, mock.py
  prompts/         every prompt in one place
  ask.py quiz.py languages.py textnorm.py store.py schemas.py config.py
frontend/          plain JS + CSS, no build step (index.html, app.js, app.css, sprites, sounds)
eval/run_eval.py   ablation harness with an independent judge
scripts/           run_local.sh, check_apis.py, init_secrets.py, smoke / contract tests
tests/             pytest suite
```

## Providers

| Service | Used for | Notes (wire formats in `backend/clients/`) |
|---|---|---|
| **Gemini** (`gemini-3.6-flash`, with a fallback list) | writing, checking, Ask Gree, quiz | `generateContent`, JSON mode for structured calls. Thought parts are skipped, and thinking tokens are counted. Calls fall back across models on 429/503/404, and a model whose daily quota is used up is skipped. OpenRouter (`OPENROUTER_API_KEY`) can be picked per episode in Settings; `LLM_PROVIDER` sets the default, and `openai` targets any OpenAI-compatible server. |
| **Sarvam** `bulbul:v4-flash` | the two voices | `dev_en_conversational` (Error606) and `simran_en_conversation` (Gree), with the language set per episode. Turns over 900 characters are split at sentence boundaries. Bodhan indic-speak (Gree = Deepika, Error606 = Amit) can be picked per episode in Settings; it uses `BODHAN_TTS_API_KEY`, or the Bodhan ASR key if that isn't set. `TTS_API_STYLE` sets the default. |
| **Bodhan** `indic-transcribe` | listen-back self-check | Paced to 14 calls per minute. If it answers in an unexpected script, the self-check stands down instead of re-recording everything. |
| **Bodhan** Indic-OCR | scanned pages and photos | One page per request. Photos are straightened, downscaled and re-encoded first. |

Every call has a timeout and retries with backoff. 429 responses honour `Retry-After` / `retryDelay`. A bad key produces a plain-language message in the UI.

## Run it

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env                 # non-secret settings: models, voices, knobs; set ACCESS_CODE to turn on the gate
python scripts/init_secrets.py       # creates ~/.config/podcast-studio/secrets.env (mode 600)
#   ...replace the DUMMY_ values there: LLM_API_KEY (or OPENROUTER_API_KEY), TTS_API_KEY, ASR_API_KEY, OCR_API_KEY
python scripts/check_apis.py         # one tiny call per API to confirm keys and voices
scripts/run_local.sh start           # background server on :8000, logs in data/server.log
scripts/run_local.sh status | stop | restart
```

Always start the server with `scripts/run_local.sh`. It loads `scripts/dev_env.sh` (thread caps for shared machines) and the secrets file. The server needs a single worker, because job state and locks live in-process.

**Keys stay out of the repo.** They live in the file named by `PODCAST_SECRETS_FILE` (default `~/.config/podcast-studio/secrets.env`). Settings load in this order, with later sources winning: repo `.env`, then the secrets file, then real environment variables. The app reports only *whether* each key is set, never its value. Each service switches from its offline fake to the real API as soon as its own key is set. While a required key is still a `DUMMY_` placeholder, the app runs in **demo mode** and shows a banner saying so.

## Tests

```bash
pytest -q                                  # 150 tests
MOCK_MODE=1 python scripts/smoke_test.py   # end to end with in-process mocks
python scripts/contract_test.py            # real client code against local fakes of Gemini / Sarvam / Bodhan
                                           # (wire formats, auth headers, 429 retry, bad keys, regeneration)
```

## Evaluation

```bash
python eval/run_eval.py --chapters chem-reactions,crop-production,nationalism-europe --length short [--audio]
EVAL_JUDGE_BASE_URL=... EVAL_JUDGE_MODEL=... python eval/run_eval.py     # optional independent judge
```

The planner runs once per chapter. Four write policies then share the same outline:

| Policy | What it does |
|---|---|
| `no_critic` | N = 1, no verifier |
| `single` | N = 1, the critic only scores |
| `uniform` | best-of-N, with the budget split evenly |
| `adaptive` | difficulty-aware allocation (ours) |

The harness reports writer calls, tokens and time; independent judge scores for faithfulness, naturalness, engagement and pedagogy; the claim-support rate; and, with `--audio`, the ASR WER before and after the self-check. Results go to `eval/results/<timestamp>/results.md`.

## Deploy

The live instance is served by `scripts/run_local.sh` behind a public URL. For a hosted setup there is a `Dockerfile` and a Render blueprint (`render.yaml`). Either way the deployment needs:
- one worker;
- a persistent disk at `DATA_DIR`;
- an always-on plan, so there is no cold start;
- `/api/health` as the health check.

All provider URLs come from environment variables (see `.env.example`).

## Known limitations

- Episodes are capped at about 2 minutes in Quick mode. Long chapters are covered by choosing sections.
- Deep agent mode writes in English only; the other 11 languages use Quick mode (the launch screen locks the language to English when Deep is picked).
- Dense derivations and tables are summarised rather than read out, and figures are not described.
- Number and symbol reading is rule-based (`backend/textnorm.py`), so rare notation may still be read literally.

## Credits

Sound effects are from Pixabay under the Pixabay Content License; see `frontend/sounds/CREDITS.md`. The bundled sample chapters are excerpts from NCERT textbooks, used for education; the original pages of each are in [`samples/textbooks/`](samples/textbooks/README.md). The Hindi and Malayalam ones were imported from an OCR'd page corpus with `scripts/import_ocr_chapters.py`.
