# Plan

## Approach

Keep the working pipeline (ingest → ReAct planner → writer/critic → claim check → TTS/ASR → stitch) and its API contracts. Rebuild the frontend as a warm, animated "study companion" for ages 10–15 (matches the Class 9–10 NCERT samples; all three seats agreed on this target). Add a few small, testable backend pieces.

1. **Secrets outside the repo.** Keys live in `~/.config/podcast-studio/secrets.env`. The path can be changed with `PODCAST_SECRETS_FILE`. `backend/config.py` loads the repo `.env` first, then the secrets file (later wins); real environment variables still override both.
   - `scripts/init_secrets.py` creates a `DUMMY_` template with mode 0600 if the file is missing, never overwrites it, and never prints its contents.
   - `/api/config` exposes only the path and a configured yes/no per service (Gemini, Sarvam, Bodhan). Key values are never returned or logged.
   - Agents do not open this file.
   - Tests point `PODCAST_SECRETS_FILE` at a nonexistent path.
2. **Kid UI: Pick → Make → Listen.**
   - **Pick:** colourful subject cards and a drag-and-drop PDF "magic box"; paste text is secondary.
   - **Choose:** length as Quick Bite / Story Time / Deep Dive, tone as Fun / Balanced / Exam Hero. These presets come from `/api/config` and map onto the existing request values.
   - **Make:** SVG hosts Riya and Arjun, with simple speaking/idle states (no lip-sync). A progress journey shows plain-language steps taken from the agent trace and an estimated wait. Queued and failed states are clear, with a retry button that uses the existing resume endpoint.
   - **Listen:** a big player and a karaoke transcript: the active line is highlighted and the active host bounces; tap a line to seek. Source badges and a "Fact-checked from your chapter" badge.
   - **Fix-it chips** ("Simpler 🐣", "Add an example 🍎", "Make it fun 🎈") use the existing one-segment regeneration; version switching is kept.
   - **Quiz** (Codex's correctness guard adopted). Fill-in-the-blank questions built from verified key facts: one key term is blanked. Distractors are terms from other facts that do not appear in the cited passage. Questions that fail this check are dropped; with fewer than 3 valid questions the quiz is hidden. No model calls.
   - **Teacher view toggle:** the agent trace, quality & compute, claim check, sources and version compare, restyled.
   - **Setup card:** shown while any key is a placeholder. It names the secrets-file path and the three keys.
3. **Frontend structure.** `frontend/index.html`, `app.css`, `app.js` and inline SVG art. No CDN, external fonts or downloaded assets. Mobile-first, keyboard-navigable, honours `prefers-reduced-motion`, dark mode. The current UI is kept as `frontend/classic.html` at `/classic`, as a reference and fallback.
4. **Reliability.** Fix the two current test failures at their root:
   - the planner test reaches a real model server in mock mode (cause not yet diagnosed; Gemini suspects the cached settings);
   - the MP3 decode test needs the `imageio-ffmpeg` dev dependency installed.
   Every error shown in the UI is a friendly message with no traceback.

## Out of scope

- Accounts/login, sharing, non-English generation, voice cloning
- New model-based features (the quiz is deterministic)
- Lip-sync animation, generated or downloaded illustrations, external fonts
- Playwright/browser end-to-end tests (would need an install)
- A pure-Python MP3 decoder
- Changes to the Dockerfile, render.yaml or deployment
- Changes to the agent's algorithms beyond bug fixes

## Acceptance criteria

1. `MOCK_MODE=1 python -m pytest -q` passes with 0 failures. No test is skipped, xfailed, deleted or weakened, and the two currently failing tests pass unchanged.
2. With `PODCAST_SECRETS_FILE` pointing at a temp file, settings take keys from that file over the repo `.env`, and a real environment variable overrides both (unit test).
3. A sentinel key loaded from the secrets file appears in no response from `/api/health`, `/api/config`, error responses or captured log output (test).
4. `scripts/init_secrets.py --path <tmp>` creates the file with mode 0600 containing `LLM_API_KEY`, `TTS_API_KEY` and `ASR_API_KEY` placeholders. A second run leaves an existing file byte-identical. Its stdout never contains the file's values (test).
5. `/api/config` returns `secrets_path` and a `configured` boolean per service. All three are false with placeholder keys and true with real-looking ones (test).
6. `/api/config` returns length and tone presets with kid labels. Every preset value is accepted by the existing episode-request schema (test).
7. `GET /` serves the new UI. `/app.css` and `/app.js` are served with CSS/JS content types. `/classic` serves the previous UI (test).
8. DOM contract: every element ID referenced in `app.js` exists in `index.html`. Every `/api/...` path in `app.js` matches a registered FastAPI route. The teacher-view panels exist: trace, quality, claims, sources, versions (static test).
9. Accessibility check: every `button`, `input`, `select` and `textarea` in `index.html` has visible text, `aria-label` or an associated `<label>`. Every SVG has a `<title>` or `aria-label`. `app.css` has a `prefers-reduced-motion` rule (static test).
10. `GET /api/episodes/{eid}/quiz`:
    - each question cites the `chunk_id` of a verified key fact, and its answer occurs in that fact;
    - no distractor equals the answer or occurs (case-insensitive) in the cited passage;
    - it returns an empty list when fewer than 3 questions are valid (tests, including a constructed ambiguous case).
11. End-to-end through FastAPI's TestClient in mock mode: sample → generate → done → episode audio served → regenerate one segment with "simpler" → new version active → switch back → quiz → transcript (test; covers the `smoke_test.py` flow).
12. An episode whose job raises ends in status `failed` with a user-readable message containing no traceback, and resume restarts it (test).
13. `ruff check backend scripts tests` reports no errors.

## Test command

`MOCK_MODE=1 python -m pytest -q`

## Allowed commands

- `pip install -r requirements-dev.txt` (installs `imageio-ffmpeg`, `pytest`, `ruff`; needs owner approval)
- `ruff check backend scripts tests`
- `MOCK_MODE=1 python scripts/smoke_test.py`
- `python scripts/init_secrets.py --path <tmp path>` in tests; the owner runs it once without `--path` to create the real template
- `MOCK_MODE=1 scripts/run_local.sh start|status|stop` for a manual look at the UI in demo mode
