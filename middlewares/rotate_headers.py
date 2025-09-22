"""Downloader middleware responsible for rotating outbound request headers."""

from __future__ import annotations

import random
from collections import deque
from typing import Deque, Dict, List

from scrapy.http import Request


class RandomHeadersMiddleware:
    """Rotate User-Agent and locale headers to reduce request fingerprinting."""

    _ACCEPT_LANGUAGES: List[str] = [
        "es-AR,es;q=0.9",
        "es-419,es;q=0.9",
        "es-ES,es;q=0.9",
    ]
    _ACCEPT_LANG_WEIGHTS: List[float] = [0.6, 0.2, 0.2]

    def __init__(self) -> None:
        chrome_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_2) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.6943.141 Safari/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.6834.201 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.6778.204 Safari/537.36",
        ]
        firefox_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:129.0) Gecko/20100101 Firefox/129.0",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 14.2; rv:128.0) Gecko/20100101 Firefox/128.0",
            "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:127.0) Gecko/20100101 Firefox/127.0",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:126.0) Gecko/20100101 Firefox/126.0",
        ]

        self._family_agents: Dict[str, List[str]] = {
            "chrome": chrome_agents,
            "firefox": firefox_agents,
        }
        for agents in self._family_agents.values():
            random.shuffle(agents)

        families = list(self._family_agents.keys())
        random.shuffle(families)
        self._family_cycle: Deque[str] = deque(families)
        self._family_indices: Dict[str, int] = {family: 0 for family in self._family_agents}

    def _next_user_agent(self) -> str:
        family = self._family_cycle[0]
        self._family_cycle.rotate(-1)

        agents = self._family_agents[family]
        index = self._family_indices[family]
        user_agent = agents[index]

        index += 1
        if index >= len(agents):
            random.shuffle(agents)
            index = 0
        self._family_indices[family] = index

        return user_agent

    def _choose_accept_language(self) -> str:
        return random.choices(self._ACCEPT_LANGUAGES, weights=self._ACCEPT_LANG_WEIGHTS, k=1)[0]

    def process_request(self, request: Request, spider):  # type: ignore[override]
        if not request.headers.get(b"User-Agent"):
            user_agent = self._next_user_agent()
            request.headers[b"User-Agent"] = user_agent.encode("utf-8")

        if not request.headers.get(b"Accept-Language"):
            accept_language = self._choose_accept_language()
            request.headers[b"Accept-Language"] = accept_language.encode("utf-8")

        referer = request.meta.get("referer")
        if referer and not request.headers.get(b"Referer"):
            request.headers[b"Referer"] = str(referer).encode("utf-8")

        return None