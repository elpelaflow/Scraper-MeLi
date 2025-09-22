+83
-19

"""Downloader middleware responsible for rotating outbound request headers."""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List

from scrapy.http import Request


@dataclass(frozen=True)
class UserAgentProfile:
    """Attributes associated with a browser user-agent profile."""

    user_agent: str
    sec_ch_ua: str | None = None
    sec_ch_ua_mobile: str | None = None
    sec_ch_ua_platform: str | None = None


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
            UserAgentProfile(
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36"
                ),
                sec_ch_ua='"Not/A)Brand";v="99", "Google Chrome";v="134", "Chromium";v="134"',
                sec_ch_ua_mobile="?0",
                sec_ch_ua_platform='"Windows"',
            ),
            UserAgentProfile(
                user_agent=(
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_2) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.6943.141 Safari/537.36"
                ),
                sec_ch_ua='"Not/A)Brand";v="99", "Google Chrome";v="133", "Chromium";v="133"',
                sec_ch_ua_mobile="?0",
                sec_ch_ua_platform='"macOS"',
            ),
            UserAgentProfile(
                user_agent=(
                    "Mozilla/5.0 (X11; Linux x86_64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.6834.201 Safari/537.36"
                ),
                sec_ch_ua='"Not/A)Brand";v="99", "Google Chrome";v="132", "Chromium";v="132"',
                sec_ch_ua_mobile="?0",
                sec_ch_ua_platform='"Linux"',
            ),
            UserAgentProfile(
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.6778.204 Safari/537.36"
                ),
                sec_ch_ua='"Not/A)Brand";v="99", "Google Chrome";v="131", "Chromium";v="131"',
                sec_ch_ua_mobile="?0",
                sec_ch_ua_platform='"Windows"',
            ),
        ]
        firefox_agents = [
            UserAgentProfile(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:129.0) Gecko/20100101 Firefox/129.0",
            ),
            UserAgentProfile(
                user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 14.2; rv:128.0) Gecko/20100101 Firefox/128.0",
            ),
            UserAgentProfile(
                user_agent="Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:127.0) Gecko/20100101 Firefox/127.0",
            ),
            UserAgentProfile(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:126.0) Gecko/20100101 Firefox/126.0",
            ),
        ]

        self._family_agents: Dict[str, List[UserAgentProfile]] = {
            "chrome": chrome_agents,
            "firefox": firefox_agents,
        }
        for agents in self._family_agents.values():
            random.shuffle(agents)

        families = list(self._family_agents.keys())
        random.shuffle(families)
        self._family_cycle: Deque[str] = deque(families)
        self._family_indices: Dict[str, int] = {family: 0 for family in self._family_agents}

    def _next_user_agent(self) -> UserAgentProfile:
        family = self._family_cycle[0]
        self._family_cycle.rotate(-1)

        agents = self._family_agents[family]
        index = self._family_indices[family]
        profile = agents[index]

        index += 1
        if index >= len(agents):
            random.shuffle(agents)
            index = 0
        self._family_indices[family] = index

        return profile

    def _choose_accept_language(self) -> str:
        return random.choices(self._ACCEPT_LANGUAGES, weights=self._ACCEPT_LANG_WEIGHTS, k=1)[0]

    def process_request(self, request: Request, spider):  # type: ignore[override]
        profile = self._next_user_agent()
        request.headers[b"User-Agent"] = profile.user_agent.encode("utf-8")

        if not request.headers.get(b"Accept-Language"):
            accept_language = self._choose_accept_language()
            request.headers[b"Accept-Language"] = accept_language.encode("utf-8")

        accept_header = request.headers.get(b"Accept", b"").lower()
        if b"text/html" in accept_header:
            if profile.sec_ch_ua:
                request.headers[b"sec-ch-ua"] = profile.sec_ch_ua.encode("utf-8")
            if profile.sec_ch_ua_mobile:
                request.headers[b"sec-ch-ua-mobile"] = profile.sec_ch_ua_mobile.encode("utf-8")
            if profile.sec_ch_ua_platform:
                request.headers[b"sec-ch-ua-platform"] = profile.sec_ch_ua_platform.encode("utf-8")

            if b"sec-fetch-dest" not in request.headers:
                request.headers[b"sec-fetch-dest"] = b"document"
            if b"sec-fetch-mode" not in request.headers:
                request.headers[b"sec-fetch-mode"] = b"navigate"
            if b"sec-fetch-site" not in request.headers:
                request.headers[b"sec-fetch-site"] = b"none"
            if b"upgrade-insecure-requests" not in request.headers:
                request.headers[b"upgrade-insecure-requests"] = b"1"

        referer = request.meta.get("referer")
        if referer and not request.headers.get(b"Referer"):
            request.headers[b"Referer"] = str(referer).encode("utf-8")

        return None