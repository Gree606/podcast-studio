import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ["MOCK_MODE"] = "1"
os.environ["ACCESS_CODE"] = ""  # never inherit the real entry code from .env
os.environ["DATA_DIR"] = tempfile.mkdtemp(prefix="podcast_tests_")
# Never read the real API-key file during tests: point it at a path that doesn't exist.
os.environ["PODCAST_SECRETS_FILE"] = os.path.join(tempfile.mkdtemp(prefix="podcast_nosecrets_"), "secrets.env")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
