# Mission 606: Earth Unknown

Turn an NCERT chapter into a two-host podcast. **Gree**, an astronaut from India, explains the chapter to **Error606**, a curious alien visiting Earth. The episode stays faithful to the book, every spoken line is fact-checked against the chapter, and you can regenerate any single part.

*Course project (Podcast Generation) for Recent Advances in Generative AI, IIT Madras, Jul–Nov 2026.*

| Spec | What this app does |
|---|---|
| **Input:** an NCERT chapter or section | A PDF (every page read with Bodhan Indic-OCR, up to 15 pages), up to 6 phone photos of pages, pasted text, or one of 9 bundled NCERT sample chapters (5 English, 2 Hindi, 2 Malayalam). Pick the sections to cover. |
| **Output:** a conversational podcast | An MP3 of about 2 minutes with a synced transcript, in any of 12 languages: English, Hindi, Hinglish, Bengali, Tamil, Telugu, Kannada, Malayalam, Marathi, Gujarati, Punjabi and Odia. |
| **Reviewer:** regenerate specific segments | Regenerate one part, optionally with an instruction such as "simpler" or "add an example". Only that part is rewritten, re-recorded and re-stitched. Every version is kept. |
| **Bar:** sounds real and stays faithful | Two distinct conversational voices, natural reactions and pauses. The fact-check badge counts the lines the chapter supports. |

## Using it

1. Open the link and give Error606 the secret code. The code is required because generating an episode spends API calls.
2. Pick a sample chapter, upload a PDF, take photos or paste text, then choose the language. **☰ Menu → ⚙️ Settings** (also reachable from the launch screen's "Studio" line) picks, for each new episode:
   - **Mode:** ⚡ Quick (default, 2 LLM calls) or 🧠 Deep agent (ReAct planner with RAG search, a critic and a targeted revision; at most 10 LLM calls, English only).
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
5. **☰ Menu → Your episodes** lists the episodes made in this browser, each tagged ⚡ Quick or 🧠 Deep. While an episode is being made, a ⏹ Stop button sits on the stage (what's done is kept; Try again continues). 🗑 deletes an episode for good, from the list or from the Save panel. The browser's Back button always returns to the home screen.

**Privacy and ownership.** There is no public list of episodes; each episode is reachable only through its link. The browser that made an episode holds an owner cookie, and only that browser can regenerate parts, switch versions or restart. Anyone with the link can listen, take the quiz and ask Gree. `/api/config` reveals no server paths or key values. Expensive calls are rate-limited per client, and repeated wrong codes are slowed down.

**Never a frozen screen.** Each LLM call times out after 45 s and falls back to the next model in the list. The progress bar keeps moving during a long call. After 100 s (10 min for Deep) the page says "Taking longer than usual" and shows a Try again button (`POST /api/episodes/{id}/restart`). Restarting the server marks running jobs as interrupted, and they can be resumed.

**Layout.** The app works on desktop (two columns), on phones down to 320 px wide, and on rotated phones. It has pixel-art sprites, an animated space scene and optional sound effects (there is a sound switch on every page).

## How it works

Both modes share the same ingestion, tools and audio stages. The mode only changes how the script is written.

```
PDF / photos / text ─► ingest.py   Bodhan Indic-OCR for every PDF page (paced; the PDF's own text as fallback)
                                   and for photos; NCERT clean-up (print codes, repeated vowel signs,
                                   faux-bold duplicates, drop caps, running heads, exercises skipped)
                     ─► sections ─► ~170-word passages ─► BM25 index (retrieval.py)

⚡ QUICK  (agent/fast.py, default: 2 calls)        🧠 DEEP AGENT  (agent/lean.py: at most 10 calls)
  Perceive  retrieve the passages in scope           Plan     ReAct planner (planner.py) with tools and RAG search
  Act       call 1: the whole episode as 3 parts,             (BM25 + embeddings); key facts pass a rule check
            each citing the passages it used                  and one LLM judge (tools.py)
  Observe   call 2: check EVERY spoken line against  Act      write all parts in one call, following the plan
            those passages; rewrite only the         Observe  critic grades every part in one call
            lines the passages don't support         Act      revise the weakest parts from the critic's note
                                                     Observe  re-grade, keep the better draft; fact-check all lines

                     ─► agent/voice.py  text normalisation (H2O, 5 m/s, years, symbols) ─► TTS per line
                                        ─► ASR transcribes its own audio ─► WER > 25% ⇒ re-record
                     ─► audio/stitch.py trim, loudness-match, conversational pauses, intro/outro, MP3 + timestamps
```

**Deep agent, step by step.** Every step shows up in the Agent panel's trace.

| # | Step | LLM calls | Course idea |
|---|---|---|---|
| 1 | Planner: Thought → Action → Observation with `list_sections`, `search` (keyword + embedding RAG), `read`, `define`, then `finish` with an outline of 3 parts, their key facts and a difficulty for each | ≤ 3 | ReAct |
| 2 | Key-fact check: rule-based (the cited passage must lexically support the fact), then one LLM judge for all facts | 1 | ToolACE dual-layer verification |
| 3 | Write every part in one call, following the plan, its verified facts and earlier lessons | 1 | |
| 4 | Critic grades every part on the 5-axis anchored rubric, plus rule checks (monologues, length, banned phrases) | 1 | ToolRL-style reward design |
| 5 | Revise only the parts below the bar, weakest and hardest first, from the critic's note stored as a lesson | ≤ 2 | Reflexion; Snell et al. (spend compute where it helps) |
| 6 | Re-grade the revisions and keep the better draft | 1 | best-of-2 with a verifier |
| 7 | Fact-check every spoken line against the passages, with fixes | 1 | |
|   | Then TTS, ASR listen-back and mixing (no LLM) | 0 | |
|   | **Total** | **≤ 10** (`DEEP_MAX_LLM_CALLS`) | |

Regenerating one part costs 2 calls (rewrite, then fact-check) in either mode.

**Why Quick is the default.** A Quick episode costs about $0.025 through OpenRouter and a Deep one about $0.06. The first Deep agent drafted, critiqued and claim-checked each part separately: 25–48 calls (about $0.36 per episode), too many for free-tier quotas and rate limits, and for a 2-minute script little better than one call to a thinking model. That per-part version (best-of-N drafts per part) is still in `search.py`/`claims.py` (`AGENT_STYLE=per_part`) for the eval ablation. The steps one call cannot do stay as tool steps: checking the script against the book, listening back to its own audio, OCR and per-part regeneration.

| Idea from the course | Where it is used |
|---|---|
| **Snell et al. 2024**, compute-optimal test-time scaling | Deep: the fixed revision budget goes to the parts the verifier scores lowest (hardest first), and a re-grade picks the better draft. Quick (2 calls) vs Deep (≤ 10) vs per-part best-of-N (25–48) is itself a compute-allocation choice, and `eval/` ablates it. |
| **ReAct** (Yao et al.) | Deep planner: an interleaved Thought / Action / Observation loop over chapter tools, visible in the Agent panel. |
| Retrieval-augmented generation | Deep agent: the planner's `search` tool and the fact-check's evidence use hybrid retrieval, keyword BM25 plus `google/gemini-embedding-001` embeddings (via OpenRouter) fused by reciprocal rank, with vectors kept in a SQLite store. Meaning search also links an English question to a Hindi or Malayalam passage. Falls back to BM25 if embeddings are unavailable. |
| **Reflexion** (Shinn et al.) | Deep agent: the critic's feedback is stored as a lesson in the part's memory and drives the revision (and later regenerations). |
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
  retrieval.py     BM25 over passages; HybridIndex = BM25 + embeddings fused by reciprocal rank (deep agent)
  vectors.py       SQLite vector store: passage/query embeddings cached by model + text hash
  providers.py     per-episode choice of script AI (Gemini / OpenRouter) and voices (Sarvam / Bodhan)
  agent/           fast.py (Quick) · lean.py (Deep: plan, write, critic, revise, fact-check in ≤ 10 calls)
                   planner.py, tools.py (ReAct + RAG search) · critic.py, writer.py · voice.py (TTS + ASR listen-back)
                   search.py, allocation.py, claims.py: the older per-part agent, kept for the eval ablation
  audio/           stitch.py, wer.py
  clients/         llm.py (Gemini / OpenRouter / OpenAI-compatible), embed.py, tts.py, asr.py, ocr.py, mock.py
  prompts/         every prompt in one place
  ask.py quiz.py languages.py textnorm.py store.py schemas.py config.py
frontend/          plain JS + CSS, no build step (index.html, app.js, app.css, sprites, sounds)
samples/           the 9 bundled chapters (Markdown) + textbooks/ (their original pages as PDFs)
eval/run_eval.py   ablation harness with an independent judge
scripts/           run_local.sh, check_apis.py, init_secrets.py, smoke / contract tests
tests/             pytest suite
```

## Providers

| Service | Used for | Notes (wire formats in `backend/clients/`) |
|---|---|---|
| **Gemini** (`gemini-3.6-flash`, with a fallback list), through **OpenRouter** by default | writing, checking, Ask Gree, quiz | OpenRouter's chat API with JSON mode, or Google's `generateContent` directly (`LLM_PROVIDER=gemini`); each can be picked per episode in Settings when its key is set. Calls fall back across models on 429/503/404, a model whose daily quota is used up is skipped, and thinking tokens are counted. `LLM_PROVIDER=openai` targets any OpenAI-compatible server. |
| **Sarvam** `bulbul:v4-flash` | the two voices | `dev_en_conversational` (Error606) and `simran_en_conversation` (Gree), with the language set per episode. Turns over 900 characters are split at sentence boundaries. Bodhan indic-speak (Gree = Deepika, Error606 = Amit) can be picked per episode in Settings; it uses `BODHAN_TTS_API_KEY`, or the Bodhan ASR key if that isn't set. `TTS_API_STYLE` sets the default. |
| **Bodhan** `indic-transcribe` | listen-back self-check | Paced to 14 calls per minute. If it answers in an unexpected script, the self-check stands down instead of re-recording everything. |
| **Bodhan** Indic-OCR | every PDF page (up to 15 pages; longer PDFs: scanned pages only) and photos | One page per request, paced at 12 pages a minute (bursts get rate-limited). A page whose OCR fails keeps the PDF's own text. NCERT print codes and the repeated vowel signs of Hindi text layers are cleaned up. Photos are straightened, downscaled and re-encoded first. |

Every call has a timeout and retries with backoff. 429 responses honour `Retry-After` / `retryDelay`. A bad key produces a plain-language message in the UI.

## Run your own copy

These steps take a fresh Linux or macOS machine to a running app. The app is one Python process
(FastAPI); there is no database server, no build step and no GPU. Episodes and audio are saved as
files under `data/`.

### 1. What you need

- **Python 3.12** (3.11 may work; the app is developed and tested on 3.12). Check with `python3.12 --version`.
  If you don't have it, [uv](https://docs.astral.sh/uv/) can fetch one: `curl -LsSf https://astral.sh/uv/install.sh | sh`.
- **git**.
- Nothing else to install: ffmpeg comes with the `imageio-ffmpeg` package, and the audio library with `soundfile`.
- **API keys** (all are paid or free-tier web APIs, no local models):

| Key | Needed for | Where to get it | Required? |
|---|---|---|---|
| `OPENROUTER_API_KEY` | writing and checking the script, Ask Gree, quiz, the deep agent's embedding search (Gemini models through OpenRouter) | https://openrouter.ai/keys | **yes** |
| `TTS_API_KEY` | the two voices (Sarvam bulbul) | https://dashboard.sarvam.ai | yes |
| `ASR_API_KEY` | listening back to its own audio (Bodhan) | Bodhan (https://bodhan.ai) | optional: without it the listen-back check is skipped |
| `OCR_API_KEY` | reading PDF pages and photos (Bodhan Indic-OCR) | Bodhan, usually the same key as ASR | optional: without it PDFs use their own text and photos can't be read |
| `LLM_API_KEY` | Google Gemini directly, as an alternative to OpenRouter (pick it in Settings, or `LLM_PROVIDER=gemini` in `.env`) | Google AI Studio: https://aistudio.google.com/apikey | optional |

With **only the OpenRouter key**, everything except the voices is real: scripts, fact-checks, Ask Gree, quiz and RAG search; the audio is a placeholder tone and the self-check is skipped. Add the Sarvam key for real voices. With no keys at all the app runs in **demo mode** (placeholder scripts and tones, with a banner saying so), which is enough to click around.

### 2. Get the code and install

```bash
git clone https://github.com/Gree606/podcast-studio.git
cd podcast-studio
python3.12 -m venv .venv            # or, with uv: uv venv --seed --python 3.12 .venv  (--seed puts pip in it)
. .venv/bin/activate
pip install -r requirements.txt     # add -dev for the tests: pip install -r requirements-dev.txt
```

### 3. Settings and keys

```bash
cp .env.example .env                # non-secret settings: models, voices, limits
python scripts/init_secrets.py      # creates ~/.config/podcast-studio/secrets.env, readable only by you
```

Open `~/.config/podcast-studio/secrets.env` and replace each `DUMMY_...` value with your key (leave optional
ones as they are, or delete those lines). Keys never go in the repo: the secrets file lives outside it, and
`.env` (git-ignored) holds only settings. To keep the secrets file somewhere else, run
`python scripts/init_secrets.py --path /your/path/secrets.env` and add `PODCAST_SECRETS_FILE=/your/path/secrets.env` to `.env`.

Worth setting in `.env`:
- `ACCESS_CODE=...`: the secret code Error606 asks for before anyone can make episodes (strongly recommended on a public link, since every episode spends your API credit). Empty means no gate.
- `DATA_DIR`: where episodes are kept (default `./data`). Back this folder up to keep episodes.

Then check the keys with one tiny call per service:

```bash
python scripts/check_apis.py
```

### 4. Start, stop, restart

```bash
scripts/run_local.sh start          # runs in the background on port 8000 (or the next free one)
scripts/run_local.sh status
scripts/run_local.sh restart        # after changing .env, the secrets file, or the code
scripts/run_local.sh stop
```

Open http://localhost:8000 (logs: `data/server.log`). On a remote server, the start command prints an
`ssh -L` line for viewing it from your laptop.

Always use `scripts/run_local.sh`: it caps thread use for shared machines, loads the secrets file, and runs
**exactly one worker** (episode jobs and their locks live in that process).

### 5. Give it a public link

- **Quickest:** a Cloudflare quick tunnel (no account needed), as used for the course demo:
  `cloudflared tunnel --url http://localhost:8000` prints an `https://...trycloudflare.com` address.
  The address changes each time the tunnel restarts. Requests are cut off after about 100 s, which is why
  PDF uploads OCR at most 15 pages.
- **Your own domain:** put nginx or Caddy in front of port 8000 (proxy everything; nothing else to configure).
- **Docker / Render:** a `Dockerfile` (Python 3.12, one worker, data in `/data`) and a Render blueprint
  (`render.yaml`) are included, but were not re-tested for this release. Pass the keys as environment
  variables, mount a persistent volume at `/data`, and use `/api/health` as the health check.

**How settings load.** The repo `.env`, then the secrets file, then real environment variables (later wins).
The app only ever reports *whether* a key is set, never its value. Each service switches from its offline
fake to the real API as soon as its own key is set.

**If something goes wrong.**
- *Demo mode banner:* `OPENROUTER_API_KEY` or `TTS_API_KEY` is still a `DUMMY_` value, or the server wasn't restarted after editing the secrets file.
- *"Port in use":* `run_local.sh start` picks the next free port and prints it; `status` shows which one.
- *Slow or "rate-limiting us" messages:* the providers limit bursts. Calls are paced and retried automatically; Sarvam voices are faster than Bodhan voices.

## Tests

```bash
pytest -q                                  # 178 tests, no network or keys needed
MOCK_MODE=1 python scripts/smoke_test.py   # end to end with in-process mocks
python scripts/contract_test.py            # real client code against local fakes of Gemini / Sarvam / Bodhan
                                           # (wire formats, auth headers, 429 retry, bad keys, regeneration)
```

## Evaluation

```bash
python eval/run_eval.py --chapters chem-reactions,crop-production,nationalism-europe --length short [--audio]
EVAL_JUDGE_BASE_URL=... EVAL_JUDGE_MODEL=... python eval/run_eval.py     # optional independent judge
```

The harness compares compute policies on the per-part agent (`AGENT_STYLE=per_part`), which spends a writer budget per part. The planner runs once per chapter. Four write policies then share the same outline:

| Policy | What it does |
|---|---|
| `no_critic` | N = 1, no verifier |
| `single` | N = 1, the critic only scores |
| `uniform` | best-of-N, with the budget split evenly |
| `adaptive` | difficulty-aware allocation (ours) |

The harness reports writer calls, tokens and time; independent judge scores for faithfulness, naturalness, engagement and pedagogy; the claim-support rate; and, with `--audio`, the ASR WER before and after the self-check. Results go to `eval/results/<timestamp>/results.md`.

## Known limitations

- Episodes are capped at about 2 minutes in Quick mode. Long chapters are covered by choosing sections.
- Deep agent mode writes in English only; the other 11 languages use Quick mode (the launch screen locks the language to English when Deep is picked).
- Dense derivations and tables are summarised rather than read out, and figures are not described.
- Number and symbol reading is rule-based (`backend/textnorm.py`), so rare notation may still be read literally.
- PDFs longer than 15 pages are OCR'd only on their scanned pages (the rest use the PDF's own text), so uploads finish within the public link's time limit.
- End-of-chapter exercises are recognised by English headings only; in Hindi and other PDFs they may be included.

## Credits

Sound effects are from Pixabay under the Pixabay Content License; see `frontend/sounds/CREDITS.md`. The bundled sample chapters are excerpts from NCERT textbooks, used for education; the original pages of each are in [`samples/textbooks/`](samples/textbooks/README.md). The Hindi and Malayalam ones were imported from an OCR'd page corpus with `scripts/import_ocr_chapters.py`.
