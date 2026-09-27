"""Stable paths shared by command-line tools and hardware libraries."""
from pathlib import Path

TOOLS_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TOOLS_ROOT.parent
THIRD_PARTY_ROOT = TOOLS_ROOT / "3rdparty"


def resolve_path(path: Path) -> Path:
    """Resolve relative inputs against the project root, independent of cwd."""
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()
