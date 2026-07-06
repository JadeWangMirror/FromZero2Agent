"""
L1 网络工具集 — web_fetch + web_search。

web_fetch   抓取 URL，返回纯文本（HTML 自动转文本），截断超长内容。
web_search  通过 DuckDuckGo（无需 API Key）搜索，返回标题+链接+摘要。
"""

from __future__ import annotations

import re
from html.parser import HTMLParser

import httpx

from tools import Tool

DEFAULT_MAX_CHARS = 20_000
USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
              "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36")


class _HTMLToText(HTMLParser):
    """极简 HTML → 纯文本转换。"""

    _SKIP = {"script", "style", "noscript", "svg", "head"}

    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in self._SKIP:
            self._skip_depth += 1
        elif tag in ("p", "br", "div", "li", "h1", "h2", "h3", "tr"):
            self._parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self._SKIP and self._skip_depth > 0:
            self._skip_depth -= 1
        elif tag in ("p", "div", "li", "h1", "h2", "h3", "tr"):
            self._parts.append("\n")

    def handle_data(self, data):
        if self._skip_depth == 0:
            self._parts.append(data)

    def get_text(self) -> str:
        text = "".join(self._parts)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()


def _html_to_text(html: str) -> str:
    parser = _HTMLToText()
    parser.feed(html)
    parser.close()
    return parser.get_text()


def create_web_tools() -> list[Tool]:

    def _web_fetch(url: str, max_chars: int = DEFAULT_MAX_CHARS) -> str:
        max_chars = max(500, min(max_chars, 200_000))
        try:
            with httpx.Client(follow_redirects=True, timeout=20.0,
                              headers={"User-Agent": USER_AGENT}) as client:
                resp = client.get(url)
                resp.raise_for_status()
        except httpx.HTTPError as e:
            return f"Error fetching {url}: {e}"

        ctype = resp.headers.get("content-type", "")
        body = resp.text
        if "html" in ctype.lower():
            text = _html_to_text(body)
        elif "json" in ctype.lower():
            text = body
        else:
            text = body
        if len(text) > max_chars:
            text = text[:max_chars] + f"\n\n... (truncated at {max_chars} chars)"
        return text or "(empty response)"

    def _web_search(query: str, max_results: int = 5) -> str:
        max_results = max(1, min(max_results, 10))
        url = "https://html.duckduckgo.com/html/"
        try:
            with httpx.Client(follow_redirects=True, timeout=20.0,
                              headers={"User-Agent": USER_AGENT}) as client:
                resp = client.post(url, data={"q": query})
                resp.raise_for_status()
        except httpx.HTTPError as e:
            return f"Error searching: {e}"

        html = resp.text
        # DuckDuckGo HTML 结果块
        results = []
        blocks = re.findall(
            r'<a[^>]+class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>'
            r'.*?<a[^>]+class="result__snippet"[^>]*>(.*?)</a>',
            html, re.DOTALL,
        )
        for href, title, snippet in blocks[:max_results]:
            # 解析 ddg 跳转链接
            m = re.search(r"uddg=([^&]+)", href)
            link = __import__("urllib.parse").unquote(m.group(1)) if m else href
            t = re.sub(r"<[^>]+>", "", title).strip()
            s = re.sub(r"<[^>]+>", "", snippet).strip()
            results.append(f"{t}\n  {link}\n  {s}")
        if not results:
            return f"(no results for '{query}')"
        return "\n\n".join(results)

    return [
        Tool("web_fetch",
             "Fetch a URL and return its text content. HTML is converted to plain text. "
             "Use for reading docs, web pages, JSON APIs.",
             {"url": {"type": "string", "description": "full URL incl. https://"},
              "max_chars": {"type": "integer", "description": f"max chars (default {DEFAULT_MAX_CHARS})"}},
             _web_fetch, required=["url"]),
        Tool("web_search",
             "Search the web via DuckDuckGo (no API key). Returns titles, links, snippets. "
             "Use when you need current info not in your training data.",
             {"query": {"type": "string"},
              "max_results": {"type": "integer", "description": "default 5, max 10"}},
             _web_search, required=["query"]),
    ]
