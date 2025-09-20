from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

_DEFAULT_QUERY_KEYS = ("query", "search_query", "last_query", "q")
_CONFIG_CANDIDATES = (
    Path("config/search_config.json"),
    Path("config/search_query.json"),
    Path("config/query.json"),
    Path("config.json"),
    Path("search_config.json"),
    Path("search_query.json"),
)


def _read_json(path: Path) -> Dict[str, Any] | str | None:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text.strip() or None


def load_search_query(default: str = "") -> str:
    """Return the last search query stored by the UI."""
    for candidate in _CONFIG_CANDIDATES:
        data = _read_json(candidate)
        if data is None:
            continue
        if isinstance(data, str):
            return data
        if isinstance(data, dict):
            for key in _DEFAULT_QUERY_KEYS:
                value = data.get(key)
                if isinstance(value, str):
                    return value
    return default
    