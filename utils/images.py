"""Shared ordering for numbered image sequences."""
from pathlib import Path


def natural_image_key(path: Path) -> tuple[int, int | str, str]:
    if path.stem.isdigit():
        return 0, int(path.stem), path.name.casefold()
    return 1, path.stem.casefold(), path.name.casefold()
