from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src", ROOT / "src" / "model", ROOT / "src" / "validated_runtime"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
