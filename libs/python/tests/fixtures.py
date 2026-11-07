import json
from pathlib import Path
from typing import Any

SPEC = Path(__file__).resolve().parents[3] / "spec" / "fixtures"


def load(name: str) -> Any:
    return json.loads((SPEC / name).read_text())
