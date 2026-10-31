"""Create the API-key file outside the repo, with placeholder values.

Run once:  python scripts/init_secrets.py            (creates ~/.config/podcast-studio/secrets.env)
           python scripts/init_secrets.py --path P  (somewhere else; then set PODCAST_SECRETS_FILE=P)

The file is created owner-only (mode 0600). An existing file is never touched,
and this script never prints the file's contents.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.config import secrets_path

TEMPLATE = """\
# Podcast Studio API keys. Keep this file private: it lives outside the repo on purpose.
# Replace each DUMMY_ value with your real key, save, then restart the app
# (scripts/run_local.sh restart). Each service switches from demo to real as soon as its key is set.

# OpenRouter (script writing, fact checks, Ask Gree, embedding search): https://openrouter.ai/keys
OPENROUTER_API_KEY=DUMMY_REPLACE_WITH_OPENROUTER_KEY

# Optional: Google Gemini directly (AI Studio, https://aistudio.google.com/apikey), picked per episode in
# Settings, or as the default with LLM_PROVIDER=gemini in .env
# LLM_API_KEY=your-gemini-key

# Sarvam AI bulbul (the two voices): https://dashboard.sarvam.ai
TTS_API_KEY=DUMMY_REPLACE_WITH_SARVAM_KEY

# Bodhan (speech recognition self-check of the audio)
ASR_API_KEY=DUMMY_REPLACE_WITH_BODHAN_KEY

# Bodhan Indic-OCR (reads PDF pages and photos; used when OCR_MODE=bodhan). Often the same key as ASR.
OCR_API_KEY=DUMMY_REPLACE_WITH_BODHAN_KEY
"""


def create(path: Path) -> bool:
    """Write the template with mode 0600. Returns False (and changes nothing) if the file exists."""
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(TEMPLATE)
    os.chmod(path, 0o600)
    return True


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--path", type=Path, help="where to create the file (default: PODCAST_SECRETS_FILE or ~/.config/podcast-studio/secrets.env)")
    args = ap.parse_args(argv)
    path = (args.path or secrets_path()).expanduser()
    if create(path):
        print(f"Created {path} (owner-only).")
        print("Open it, replace the DUMMY_ values with your keys, then restart the app.")
    else:
        print(f"{path} already exists; left it unchanged.")
    if args.path and args.path.expanduser() != secrets_path():
        print(f"Tell the app where it is: export PODCAST_SECRETS_FILE={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
