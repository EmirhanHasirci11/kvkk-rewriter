import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import kvkk_rag  # noqa: E402,F401  (puts the submodule's scripts/ on sys.path, after ours)
