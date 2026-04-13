"""Resolve layout / image / annotation paths for template benchmark JSON files."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Union


def parse_data_root(
    raw: Optional[str],
    dataset_root: Union[str, Path],
) -> Optional[Path]:
    """Resolve ``data_root`` stored in a benchmark JSON file.

    *raw* is resolved against *dataset_root* (e.g.
    ``"lica-data"`` → ``dataset_root / "lica-data"``).
    """
    if not raw:
        return None
    root = Path(raw)
    if root.is_absolute():
        return root
    return (Path(dataset_root) / root).resolve()


def resolve_layout_paths(
    layout_id: str,
    layout_index: Dict[str, str],
    data_root: Optional[Path],
) -> Dict[str, str]:
    """Map a layout ID to paths under a Lica core tree (``lica-data/``)."""
    template_id = layout_index.get(layout_id, "")
    if not data_root or not template_id:
        return {}
    return {
        "layout_path": str(data_root / "layouts" / template_id / f"{layout_id}.json"),
        "image_path": str(data_root / "images" / template_id / f"{layout_id}.png"),
        "annotation_path": str(
            data_root / "annotations" / template_id / f"{layout_id}.json"
        ),
    }


def load_layout_content(layout_id: str, paths: Dict[str, str]) -> str:
    """Read layout JSON if the file exists, otherwise return the bare ID."""
    lpath = paths.get("layout_path", "")
    if lpath and Path(lpath).is_file():
        return Path(lpath).read_text(encoding="utf-8")
    return layout_id
