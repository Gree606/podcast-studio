import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ["MOCK_MODE"] = "1"
os.environ["DATA_DIR"] = tempfile.mkdtemp(prefix="podcast_tests_")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
