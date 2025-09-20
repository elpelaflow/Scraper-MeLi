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
            value = data.strip()
            if value:
                return value
        if isinstance(data, dict):
            for key in _DEFAULT_QUERY_KEYS:
                value = data.get(key)
                if isinstance(value, str):
                    value = value.strip()
                    if value:
                        return value
    return default


def save_search_query(query: str, path: Path | None = None) -> None:
    """Persist the last search query so the dashboard can reuse it."""

    query = (query or "").strip()
    if not query:
        return

    target = Path(path) if path is not None else _resolve_target_path()

    try:
        target.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        return

    existing = _read_json(target)
    payload: Dict[str, Any]

    if isinstance(existing, dict):
        payload = dict(existing)
        payload["query"] = query
    else:
        payload = {"query": query}

    try:
        target.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except OSError:
        return


def _resolve_target_path() -> Path:
    for candidate in _CONFIG_CANDIDATES:
        candidate = Path(candidate)
        if candidate.exists():
            return candidate
    # Prefer a dedicated file for the query if nothing exists yet.
    if len(_CONFIG_CANDIDATES) > 1:
        return Path(_CONFIG_CANDIDATES[1])
    return Path(_CONFIG_CANDIDATES[0])
    