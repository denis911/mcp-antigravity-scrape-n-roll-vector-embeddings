"""
japan_scraper.py — Japanese Job Board Scraper
==============================================
Searches Japanese bilingual job boards for roles where Japanese companies
are seeking EMEA/European channel, sales, or partnerships talent.

Supported boards (DEFAULT_JAPAN_BOARDS):
  - daijob.com        — largest bilingual job board, strong international sales coverage
  - gaijinpot.com     — English-friendly, international sales exec roles
  - careercross.com   — bilingual professionals, international scope
  - jp.japanese-jobs.com — targets bilinguals for Japan-based + overseas roles

Key difference from serper_scraper.py:
  Location is NOT appended to search queries — the boards are Japan-specific by
  definition. We search for European-scope keywords (e.g. "EMEA sales") without
  a city qualifier.

Usage (via MCP tool):
    scrape_jobs(
        keywords=["EMEA sales", "Europe channel partner", "channel manager Europe"],
        locations=["Japan"],        # triggers japan_scraper routing
        job_domain="sales",
        max_total_queries=6
    )
"""

import os
import asyncio
import logging
import requests
import aiohttp
from bs4 import BeautifulSoup
import pandas as pd

from extractor import extract_structured_data_async, detect_domain, ensure_canonical_columns

SERPER_URL = "https://google.serper.dev/search"

# ── Default Japanese Job Boards ───────────────────────────────────────────────

DEFAULT_JAPAN_BOARDS = [
    "daijob.com",           # largest bilingual board; strong intl sales coverage
    "gaijinpot.com",        # English-friendly; international sales exec roles
    "careercross.com",      # bilingual professionals; intl scope
    "jp.japanese-jobs.com", # bilinguals for Japan-based + overseas roles
]

# Location strings that trigger Japan routing in job_matcher_mcp.py
DEFAULT_JAPAN_LOCATIONS = [
    "Japan",
    "Tokyo",
    "Japanese companies",
    "Japan remote",
]

# ── Query Building ────────────────────────────────────────────────────────────

def build_japan_queries(
    keywords: list[str],
    boards: list[str],
) -> list[str]:
    """
    Build Serper search queries for Japanese job boards.

    IMPORTANT: Location is intentionally NOT included in queries.
    These boards are Japan-specific. We search for European-scope
    keywords (e.g. "EMEA sales") without a city qualifier.

    Example output:
        (site:daijob.com OR site:gaijinpot.com) "EMEA sales"
        (site:daijob.com OR site:gaijinpot.com) "Europe channel partner"
    """
    domains_str = " OR ".join([f"site:{d}" for d in boards])
    footprint = f"({domains_str})"

    queries = []
    for kw in keywords:
        queries.append(f'{footprint} "{kw}"')

    return queries


# ── Serper Search ─────────────────────────────────────────────────────────────

def _serper_search_sync(query: str) -> list[dict]:
    """Synchronous Serper API call. Runs in executor to stay async-friendly."""
    api_key = os.environ.get("SERPER_API_KEY")
    if not api_key:
        raise ValueError(
            "SERPER_API_KEY environment variable is not set. "
            "Add it to claude_desktop_config.json env block."
        )

    headers = {
        "X-API-KEY": api_key,
        "Content-Type": "application/json",
    }
    payload = {"q": query, "num": 20}

    try:
        response = requests.post(SERPER_URL, headers=headers, json=payload, timeout=15)
        response.raise_for_status()
        return response.json().get("organic", [])
    except Exception as e:
        logging.error(f"Japan scraper: Serper error for '{query}': {e}")
        return []


async def search_japan_boards(queries: list[str]) -> list[dict]:
    """Run all Serper queries concurrently and deduplicate by URL."""
    loop = asyncio.get_event_loop()

    tasks = [
        loop.run_in_executor(None, _serper_search_sync, q)
        for q in queries
    ]

    results = await asyncio.gather(*tasks)

    seen_urls = set()
    deduped = []
    for batch in results:
        for item in batch:
            url = item.get("link")
            if url and url not in seen_urls:
                seen_urls.add(url)
                deduped.append({
                    "title": item.get("title", ""),
                    "url": url,
                    "company": "",          # extracted by LLM in extractor.py
                    "location": "",         # extracted by LLM
                    "description_snippet": item.get("snippet", ""),
                    "source": "japan_boards",
                })

    logging.info(f"Japan scraper: {len(deduped)} unique results from {len(queries)} queries")
    return deduped


# ── HTML Fetching ─────────────────────────────────────────────────────────────

async def fetch_job_html(url: str, session: aiohttp.ClientSession) -> str:
    """
    Fetch and clean HTML from a job posting URL.
    Returns plain text suitable for LLM extraction.
    """
    try:
        async with session.get(
            url,
            timeout=aiohttp.ClientTimeout(total=12),
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
        ) as response:
            if response.status == 200:
                html = await response.text()
                soup = BeautifulSoup(html, "html.parser")
                for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
                    tag.extract()
                return soup.get_text(separator=" ", strip=True)
            else:
                logging.warning(f"Japan scraper: HTTP {response.status} for {url}")
                return ""
    except Exception as e:
        logging.warning(f"Japan scraper: fetch error for {url}: {e}")
        return ""


async def fetch_all_descriptions(results: list[dict]) -> list[dict]:
    """Concurrently fetch full job descriptions for all results."""
    connector = aiohttp.TCPConnector(limit=30)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [fetch_job_html(item["url"], session) for item in results]
        html_texts = await asyncio.gather(*tasks)

    enriched = []
    for item, html_text in zip(results, html_texts):
        item["description_text"] = html_text if len(html_text) > 100 else item["description_snippet"]
        item["description"] = item["description_text"]
        enriched.append(item)

    return enriched


# ── Main Entry Point ──────────────────────────────────────────────────────────

async def scrape_japan(
    keywords: list[str],
    locations: list[str] | None = None,    # accepted for interface consistency, ignored
    max_per_query: int = 100,              # accepted for interface consistency
    job_domain: str | None = None,
    boards: list[str] | None = None,
) -> pd.DataFrame:
    """
    Scrape Japanese job boards for EMEA/European-scope roles at Japanese companies.

    Interface matches scrape_builtin() and scrape_serper() for clean integration
    with job_matcher_mcp.py.

    Parameters:
        keywords    : Search keywords e.g. ["EMEA sales", "Europe channel partner"]
        locations   : Accepted but ignored — boards are Japan-specific by definition
        max_per_query: Accepted but ignored — Serper handles result count
        job_domain  : "sales", "gtm", "data", "biotech", "any" (auto-detected if None)
        boards      : Override DEFAULT_JAPAN_BOARDS if needed

    Returns:
        pd.DataFrame with canonical schema columns (see TASK.md)
    """
    job_domain = job_domain or detect_domain(keywords)
    boards = boards or DEFAULT_JAPAN_BOARDS

    if locations:
        logging.info(
            f"Japan scraper: 'locations' parameter ({locations}) is ignored — "
            f"boards are Japan-specific. Searching keywords only."
        )

    logging.info(f"Japan scraper: keywords={keywords}, boards={boards}, domain={job_domain}")

    # Stage 1: Build queries and search
    queries = build_japan_queries(keywords, boards)
    logging.info(f"Japan scraper: {len(queries)} queries generated")

    organic_results = await search_japan_boards(queries)

    if not organic_results:
        logging.info("Japan scraper: no results found")
        return ensure_canonical_columns(pd.DataFrame())

    # Stage 2: Fetch full descriptions
    logging.info(f"Japan scraper: fetching {len(organic_results)} job pages...")
    enriched = await fetch_all_descriptions(organic_results)

    # Stage 3: LLM extraction (same extractor.py as other scrapers)
    logging.info("Japan scraper: running LLM extraction...")
    structured = await extract_structured_data_async(enriched, domain=job_domain)

    df = pd.DataFrame(structured)
    df = ensure_canonical_columns(df)

    logging.info(f"Japan scraper: returning {len(df)} structured records")
    return df
