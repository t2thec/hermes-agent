"""Crawl4AI self-hosted web search + extract plugin.

Zero-config local backend: uses httpx + trafilatura for static HTML,
Crawl4AI (Playwright) for JS-rendered pages, and DuckDuckGo HTML scraping
for search. No API key needed — designed as the primary backend in a
chain before falling through to paid providers (Firecrawl, Tavily, Exa).
"""

from __future__ import annotations

from plugins.web.crawl4ai.provider import Crawl4AIWebSearchProvider


def register(ctx) -> None:
    """Register the Crawl4AI provider with the plugin context."""
    ctx.register_web_search_provider(Crawl4AIWebSearchProvider())