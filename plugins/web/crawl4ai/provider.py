"""Crawl4AI self-hosted web search + extract — plugin form.

A zero-config local backend that replaces paid SaaS providers for the
~80% of web requests that hit static or lightly-JS pages.

**Search**: DuckDuckGo HTML scrape (no API key, no ddgs package needed).
**Extract**: Two-tier:
  1. Fast path — ``httpx`` + ``trafilatura`` for static HTML (~300ms).
  2. Slow path — ``crawl4ai`` (Playwright under the hood) for JS-heavy
     pages that return empty/minimal content from the fast path (~5s).
**Crawl**: Seed URL + link walking via same two-tier extract.

When this provider fails (Cloudflare block, 403, truly empty page),
the chain executor in ``web_search_registry`` falls through to the next
provider (typically Firecrawl), so the agent always gets an answer.

Config keys this provider responds to::

    web:
      search_backend: "crawl4ai"      # explicit per-capability
      extract_backend: "crawl4ai"     # explicit per-capability
      backend: "crawl4ai"             # shared fallback

Env vars::

    CRAWL4AI_PLAYWRIGHT=1             # Force Playwright for all extracts
                                      # (default: only on fast-path failure)

No API key required — this is a self-hosted backend.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any, Dict, List, Optional
from urllib.parse import quote_plus, urljoin, urlparse

from agent.web_search_provider import WebSearchProvider
from tools.url_safety import is_safe_url
from tools.website_policy import check_website_access

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy dependency probes
# ---------------------------------------------------------------------------

_httpx_available: Optional[bool] = None
_trafilatura_available: Optional[bool] = None
_crawl4ai_available: Optional[bool] = None


def _check_httpx() -> bool:
    global _httpx_available
    if _httpx_available is None:
        try:
            import httpx  # noqa: F401
            _httpx_available = True
        except ImportError:
            _httpx_available = False
    return _httpx_available


def _check_trafilatura() -> bool:
    global _trafilatura_available
    if _trafilatura_available is None:
        try:
            import trafilatura  # noqa: F401
            _trafilatura_available = True
        except ImportError:
            _trafilatura_available = False
    return _trafilatura_available


def _check_crawl4ai() -> bool:
    global _crawl4ai_available
    if _crawl4ai_available is None:
        try:
            from crawl4ai import AsyncWebCrawler  # noqa: F401
            _crawl4ai_available = True
        except ImportError:
            _crawl4ai_available = False
    return _crawl4ai_available


# ---------------------------------------------------------------------------
# Fast-path extract: httpx + trafilatura
# ---------------------------------------------------------------------------

# Common "empty page" indicators that mean the fast path failed
_JS_SHELL_PATTERNS = [
    re.compile(r"<body[^>]*>\s*(<(?:div|noscript|i?frame)[^>]*>\s*)*</body>", re.I),
    re.compile(r'<body[^>]*>\s*<div[^>]*id=["\']root["\'][^>]*>\s*</div>\s*</body>', re.I),
    re.compile(r'<body[^>]*>\s*<div[^>]*id=["\']app["\'][^>]*>\s*</div>\s*</body>', re.I),
]

_MIN_CONTENT_LENGTH = 200  # Below this, we consider the fast path failed


def _looks_like_js_shell(html: str) -> bool:
    """Return True if the HTML looks like a JS-only SPA shell with no content."""
    if not html or len(html) < 100:
        return True
    for pattern in _JS_SHELL_PATTERNS:
        if pattern.search(html):
            return True
    return False


def _extract_with_httpx(url: str) -> Optional[Dict[str, Any]]:
    """Fast-path extraction using httpx + trafilatura.

    Returns a result dict (url, title, content, raw_content, metadata)
    on success, None on failure or JS-shell detection.
    """
    import httpx

    try:
        with httpx.Client(
            follow_redirects=True,
            timeout=15.0,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/137.0.0.0 Safari/537.36"
                ),
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
            },
        ) as client:
            resp = client.get(url)
            resp.raise_for_status()

            content_type = resp.headers.get("content-type", "")
            if "text/html" not in content_type and "application/xhtml" not in content_type:
                # Not HTML — return as-is if it's text-like
                if "text/" in content_type or "json" in content_type:
                    raw = resp.text
                    return {
                        "url": str(resp.url),
                        "title": "",
                        "content": raw[:5000],
                        "raw_content": raw[:5000],
                        "metadata": {"source": "crawl4ai-httpx", "content_type": content_type},
                    }
                return None

            html = resp.text
            final_url = str(resp.url)

            # Check if this is a JS-only shell
            if _looks_like_js_shell(html):
                logger.debug("crawl4ai-httpx: JS shell detected for %s", url)
                return None

            # Extract with trafilatura — use bare_extraction for metadata + markdown
            import trafilatura

            doc = trafilatura.bare_extraction(html, include_links=True, include_tables=True, favor_precision=False)
            if doc is None:
                # Fallback: try plain text extraction
                content = trafilatura.extract(html, output_format="txt")
                if not content or len(content.strip()) < _MIN_CONTENT_LENGTH:
                    logger.debug("crawl4ai-httpx: trafilatura returned empty for %s", url)
                    return None
                return {
                    "url": final_url,
                    "title": "",
                    "content": content,
                    "raw_content": content,
                    "metadata": {"source": "crawl4ai-httpx", "content_type": content_type},
                }

            title = getattr(doc, "title", "") or ""
            content = getattr(doc, "text", "") or ""

            # Also get markdown for richer output (links, formatting)
            markdown = trafilatura.extract(html, output_format="markdown", include_links=True, include_tables=True)
            if markdown and len(markdown) > len(content):
                content = markdown

            # Fall back to HTML <title> when trafilatura doesn't extract one
            if not title:
                html_title = re.search(r"<title>([^<]+)</title>", html, re.I)
                if html_title:
                    title = html_title.group(1).strip()
                    # Decode common HTML entities
                    title = title.replace("&#8212;", "—").replace("&#8211;", "–").replace("&amp;", "&").replace("&#39;", "'")

            if not content or len(content.strip()) < _MIN_CONTENT_LENGTH:
                logger.debug("crawl4ai-httpx: content too short for %s (%d chars)", url, len(content or ""))
                return None

            return {
                "url": final_url,
                "title": title,
                "content": content,
                "raw_content": content,
                "metadata": {
                    "source": "crawl4ai-httpx",
                    "content_type": content_type,
                },
            }

    except Exception as exc:
        logger.debug("crawl4ai-httpx: fetch failed for %s: %s", url, exc)
        return None


# ---------------------------------------------------------------------------
# Slow-path extract: crawl4ai (Playwright)
# ---------------------------------------------------------------------------

async def _extract_with_crawl4ai(url: str) -> Optional[Dict[str, Any]]:
    """Slow-path extraction using crawl4ai's Playwright-backed AsyncWebCrawler.

    Returns a result dict on success, None on failure.
    """
    try:
        from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig

        browser_config = BrowserConfig(
            headless=True,
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/137.0.0.0 Safari/537.36"
            ),
        )
        run_config = CrawlerRunConfig(
            word_count_threshold=10,
            exclude_all_images=True,
        )

        async with AsyncWebCrawler(config=browser_config) as crawler:
            result = await asyncio.wait_for(
                crawler.arun(url=url, config=run_config),
                timeout=30,
            )

            if not result or not result.success:
                logger.debug("crawl4ai-playwright: failed for %s: %s", url, getattr(result, "error_message", "unknown"))
                return None

            markdown = getattr(result, "markdown", "") or ""
            if not markdown or len(markdown.strip()) < _MIN_CONTENT_LENGTH:
                logger.debug("crawl4ai-playwright: empty content for %s", url)
                return None

            title = getattr(result, "metadata", {}).get("title", "") if hasattr(result, "metadata") and isinstance(result.metadata, dict) else ""

            return {
                "url": url,
                "title": title,
                "content": markdown,
                "raw_content": markdown,
                "metadata": {
                    "source": "crawl4ai-playwright",
                },
            }

    except ImportError:
        logger.debug("crawl4ai-playwright: crawl4ai not installed")
        return None
    except asyncio.TimeoutError:
        logger.warning("crawl4ai-playwright: timed out for %s", url)
        return None
    except Exception as exc:
        logger.debug("crawl4ai-playwright: failed for %s: %s", url, exc)
        return None


# ---------------------------------------------------------------------------
# DuckDuckGo HTML search (no API key, no ddgs package)
# ---------------------------------------------------------------------------

def _ddg_html_search(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    """Scrape DuckDuckGo HTML for search results. No API key required.

    Sends a single GET to https://html.duckduckgo.com/html/ which returns
    a simple HTML page with result links. This is the same endpoint the
    ``ddgs`` Python package scrapes internally.
    """
    import httpx
    from urllib.parse import unquote

    results: List[Dict[str, Any]] = []
    try:
        with httpx.Client(
            follow_redirects=True,
            timeout=10.0,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/137.0.0.0 Safari/537.36"
                ),
            },
        ) as client:
            resp = client.get(
                "https://html.duckduckgo.com/html/",
                params={"q": query},
            )
            resp.raise_for_status()
            html = resp.text

            # Split by result__a anchor to get per-result chunks.
            # Each chunk starts with href="..." ...>Title</a>...snippet...
            chunks = re.split(r'<a[^>]+class="result__a"', html)

            position = 0
            for chunk in chunks[1:]:  # skip everything before first result
                # Title: immediately after the split point
                title_match = re.search(r"^[^>]*>([^<]+)", chunk)
                # URL: from the href attribute in the pre-split anchor
                href_match = re.search(
                    r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>' + re.escape(title_match.group(1) if title_match else ""),
                    html,
                )

                # Snippet: result__snippet in same chunk
                snippet_match = re.search(
                    r'class="result__snippet"[^>]*>([^<]+)', chunk
                )

                if not title_match:
                    continue

                title = title_match.group(1).strip().replace("&amp;", "&")

                # Extract real URL from uddg param in the href
                # Find the href that belongs to this result by scanning
                # the full HTML for this title
                real_url = ""
                url_pattern = re.compile(
                    r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>'
                    + re.escape(title_match.group(1).strip()),
                )
                url_match = url_pattern.search(html)
                if url_match:
                    raw_href = url_match.group(1)
                    uddg_match = re.search(r"uddg=([^&]+)", raw_href)
                    if uddg_match:
                        real_url = unquote(unquote(uddg_match.group(1)))
                    else:
                        real_url = raw_href

                # Skip ads and DDG internal pages
                if not real_url or "duckduckgo.com/y.js" in real_url or "ad_provider=" in real_url:
                    continue
                if "duckduckgo.com" in real_url and "uddg=" not in (url_match.group(1) if url_match else ""):
                    continue

                description = ""
                if snippet_match:
                    description = snippet_match.group(1).strip()
                    description = description.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")

                position += 1
                if position > limit:
                    break

                results.append({
                    "title": title,
                    "url": real_url,
                    "description": description,
                    "position": position,
                })

    except Exception as exc:
        logger.debug("crawl4ai ddg-search: failed: %s", exc)

    return results


# ---------------------------------------------------------------------------
# Provider class
# ---------------------------------------------------------------------------


class Crawl4AIWebSearchProvider(WebSearchProvider):
    """Self-hosted web search + extract provider. Zero API key required.

    **Search**: DuckDuckGo HTML scraping — fast, free, no key.
    **Extract**: Two-tier — httpx+trafilatura (fast) then crawl4ai/Playwright.
    **Crawl**: Seed URL + internal-link walking using the same two-tier extract.
    """

    @property
    def name(self) -> str:
        return "crawl4ai"

    @property
    def display_name(self) -> str:
        return "Crawl4AI (Self-Hosted)"

    def is_available(self) -> bool:
        """Return True when the core deps (httpx + trafilatura) are importable.

        The ``ddgs`` package is optional — if installed, search uses it
        (better CAPTCHA handling); otherwise falls back to raw DDG HTML
        scraping. The extract path (httpx + trafilatura) is always required.
        """
        return _check_httpx() and _check_trafilatura()

    def supports_search(self) -> bool:
        return True

    def supports_extract(self) -> bool:
        return True

    def supports_crawl(self) -> bool:
        return True

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": self.display_name,
            "badge": "free",
            "tag": "No API key required — self-hosted. Uses ddgs package for search (install with pip install ddgs) and httpx + trafilatura for extraction.",
            "env_vars": [],
        }

    # --- Search ----------------------------------------------------------------

    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """Search via DuckDuckGo. Uses the ``ddgs`` package if installed
        (handles CAPTCHAs and session management internally), otherwise
        falls back to raw HTML scraping of DDG's HTML endpoint.
        """
        from tools.interrupt import is_interrupted

        if is_interrupted():
            return {"success": False, "error": "Interrupted"}

        logger.info("Crawl4AI search: '%s' (limit=%d)", query, limit)

        # Prefer ddgs package — it handles DDG's anti-bot internally
        results = self._search_via_ddgs(query, limit)
        if results is not None:
            if not results:
                return {"success": False, "error": "No search results found"}
            logger.info("Crawl4AI: found %d search results (via ddgs package)", len(results))
            return {"success": True, "data": {"web": results}}

        # Fallback: raw DDG HTML scraping (less reliable, gets CAPTCHA'd)
        results = _ddg_html_search(query, limit)
        if not results:
            return {"success": False, "error": "No search results found"}

        logger.info("Crawl4AI: found %d search results (via DDG HTML scrape)", len(results))
        return {"success": True, "data": {"web": results}}

    def _search_via_ddgs(self, query: str, limit: int) -> Optional[List[Dict[str, Any]]]:
        """Search using the ``ddgs`` package. Returns None if not installed."""
        try:
            from ddgs import DDGS
        except ImportError:
            logger.debug("crawl4ai: ddgs package not installed, falling back to HTML scraping")
            return None

        try:
            web_results: List[Dict[str, Any]] = []
            with DDGS() as client:
                for i, hit in enumerate(client.text(query, max_results=limit)):
                    if i >= limit:
                        break
                    url = str(hit.get("href") or hit.get("url") or "")
                    web_results.append({
                        "title": str(hit.get("title", "")),
                        "url": url,
                        "description": str(hit.get("body", "")),
                        "position": i + 1,
                    })
            return web_results
        except Exception as exc:
            logger.warning("Crawl4AI ddgs search error: %s", exc)
            # Return empty list (not None) so caller knows ddgs was tried but failed
            return []

    # --- Extract ---------------------------------------------------------------

    async def extract(self, urls: List[str], **kwargs: Any) -> List[Dict[str, Any]]:
        """Extract content from URLs. Two-tier: httpx fast path, then crawl4ai.

        For each URL:
        1. Try httpx + trafilatura (~300ms for static pages).
        2. If empty/JS-shell detected AND crawl4ai is installed, try Playwright (~5s).
        3. If both fail, return an error entry so the chain executor can try the next provider.
        """
        from tools.interrupt import is_interrupted as _is_interrupted

        format_ = kwargs.get("format")
        force_playwright = _should_force_playwright()

        results: List[Dict[str, Any]] = []

        for url in urls:
            if _is_interrupted():
                results.append({"url": url, "error": "Interrupted", "title": ""})
                continue

            # Website policy gate
            blocked = check_website_access(url)
            if blocked:
                logger.info("Blocked extract for %s by rule %s", blocked["host"], blocked["rule"])
                results.append({
                    "url": url,
                    "title": "",
                    "content": "",
                    "error": blocked["message"],
                    "blocked_by_policy": {
                        "host": blocked["host"],
                        "rule": blocked["rule"],
                        "source": blocked["source"],
                    },
                })
                continue

            # SSRF re-check after potential redirect
            if not is_safe_url(url):
                results.append({
                    "url": url,
                    "title": "",
                    "content": "",
                    "error": "Blocked: URL targets a private or internal network address",
                })
                continue

            extracted: Optional[Dict[str, Any]] = None

            # Tier 1: httpx + trafilatura (fast)
            if not force_playwright:
                logger.info("Crawl4AI-httpx extracting: %s", url)
                extracted = await asyncio.to_thread(_extract_with_httpx, url)

            # Tier 2: crawl4ai / Playwright (slow, but handles JS)
            if extracted is None and _check_crawl4ai():
                logger.info("Crawl4AI-playwright extracting: %s", url)
                extracted = await _extract_with_crawl4ai(url)

            if extracted is not None:
                results.append(extracted)
            else:
                # Both tiers failed — return error so the chain executor
                # can try the next provider (Firecrawl, Tavily, etc.)
                logger.warning("Crawl4AI: both tiers failed for %s", url)
                results.append({
                    "url": url,
                    "title": "",
                    "content": "",
                    "error": "crawl4ai: content extraction failed (page may require anti-bot bypass)",
                })

        return results

    # --- Crawl -----------------------------------------------------------------

    async def crawl(self, url: str, **kwargs: Any) -> Any:
        """Crawl a seed URL and extract content from linked pages.

        Simple BFS: extract the seed, find internal links, extract those.
        Depth 1 only — we don't walk further than the seed's direct links.
        """
        from tools.interrupt import is_interrupted as _is_interrupted

        if _is_interrupted():
            return {"success": False, "error": "Interrupted"}

        # Extract the seed page first
        seed_results = await self.extract([url], **kwargs)
        if not seed_results or seed_results[0].get("error"):
            return {"success": False, "error": f"Crawl seed failed: {seed_results[0].get('error', 'unknown')}"}

        # Find internal links from the seed content
        seed_content = seed_results[0].get("raw_content", "") or seed_results[0].get("content", "")
        seed_title = seed_results[0].get("title", "")
        internal_links = _find_internal_links(url, seed_content, max_links=5)

        if not internal_links:
            return {"success": True, "results": seed_results}

        # Extract linked pages
        linked_results = await self.extract(internal_links, **kwargs)

        # Combine, dedup by URL
        all_results = seed_results + linked_results
        seen = set()
        deduped = []
        for r in all_results:
            r_url = r.get("url", "")
            if r_url and r_url not in seen and not r.get("error"):
                seen.add(r_url)
                deduped.append(r)

        return {"success": True, "results": deduped}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _should_force_playwright() -> bool:
    """Check if CRAWL4AI_PLAYWRIGHT env var is set to force Playwright for all extracts."""
    import os
    return os.getenv("CRAWL4AI_PLAYWRIGHT", "").strip().lower() in ("1", "true", "yes")


def _find_internal_links(base_url: str, content: str, max_links: int = 5) -> List[str]:
    """Extract same-domain links from markdown content."""
    parsed = urlparse(base_url)
    base_domain = parsed.netloc

    # Match markdown links and raw URLs
    link_re = re.compile(r'\[(?:[^\]]+)\]\(([^)]+)\)|(?<!\()https?://[^\s\)<"\']+', re.I)
    links: List[str] = []

    for match in link_re.finditer(content):
        raw = match.group(1) or match.group(0)
        # Resolve relative URLs
        full_url = urljoin(base_url, raw)
        link_parsed = urlparse(full_url)

        # Same domain only, no anchors, no file downloads
        if link_parsed.netloc != base_domain:
            continue
        if link_parsed.fragment:
            full_url = full_url.split("#")[0]
        if any(ext in link_parsed.path.lower() for ext in (".pdf", ".zip", ".png", ".jpg", ".gif", ".svg")):
            continue

        if full_url not in links and full_url != base_url:
            links.append(full_url)
        if len(links) >= max_links:
            break

    return links