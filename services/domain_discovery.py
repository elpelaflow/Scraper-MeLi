from __future__ import annotations
from typing import Any, Dict, List, Tuple
import requests

DEFAULT_SITE = "MLA"

def fetch_domain_discovery(
    query: str, limit: int = 5, site: str = DEFAULT_SITE
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Devuelve una tupla con la lista de resultados parseados y el cuerpo crudo
    del endpoint de Domain Discovery. Si falla, retorna ([], "") sin romper la UI.
    """
    if not query:
        return [], ""
    url = f"https://api.mercadolibre.com/sites/{site}/domain_discovery/search"
    try:
        resp = requests.get(url, params={"q": query, "limit": int(limit)}, timeout=10)
        resp.raise_for_status()
        text = resp.text
        data = resp.json()
        if isinstance(data, list):
            return data, text
        if isinstance(data, dict) and isinstance(data.get("results"), list):
            return data["results"], text
        return [], text
    except Exception:
        return [], ""