"""Text and typography metrics."""


def normalize_font_name(raw: str) -> str:
    """Lower-case, strip whitespace and common suffixes."""
    name = raw.strip().lower()
    for suffix in ("-regular", " regular", "-bold", " bold"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name
