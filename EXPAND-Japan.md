# EXPAND-Japan.md — Japanese Job Board Scraper

## Goal

Add a third scraper `japan_scraper.py` targeting Japanese job boards for roles where
Japanese companies are seeking EMEA/European channel, sales, or partnerships talent.
The existing `builtin_scraper.py` and `serper_scraper.py` remain untouched.

---

## Why a Separate File

`serper_scraper.py` is battle-tested and handles EU/global job boards correctly.
Its `build_serper_queries()` function always appends a location string to queries —
correct behaviour for boards like Greenhouse, Lever, and Welcome to the Jungle,
but wrong for Japanese job boards where:

1. The boards themselves are Japan-specific — no location filter needed
2. The boards use Japanese or English keywords without city-name qualifiers
3. We want to search for European-scope roles posted by Japanese companies
   (e.g. "EMEA sales", "Europe channel partner") not roles located in Japan

Keeping Japan logic in a dedicated `japan_scraper.py` gives us:
- Zero risk of breaking EU Serper behaviour
- Clean, inspectable logic per board
- Easy to add new Japanese boards later
- Independent testing

---

## Architecture Overview

```
scrape_jobs() MCP tool
        │
        ├── us_locations  → builtin_scraper.scrape_builtin()    [existing]
        ├── eu_locations  → serper_scraper.scrape_serper()      [existing]
        └── jp_locations  → japan_scraper.scrape_japan()        [NEW]
```

Japan routing trigger: any location in `DEFAULT_JAPAN_LOCATIONS` list
(e.g. "Japan", "Tokyo", "Japanese companies") routes to `japan_scraper.py`.

---

## File to Create: `japan_scraper.py`

Place in same directory as `serper_scraper.py` and `builtin_scraper.py`.

```python
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
```

---

## Changes to `job_matcher_mcp.py`

### 1. Add import

```python
# After existing scraper imports:
from serper_scraper import scrape_serper
from japan_scraper import scrape_japan, DEFAULT_JAPAN_LOCATIONS    # NEW
```

### 2. Update `DEFAULT_SOURCE_MAP` in `builtin_scraper.py`

No change needed to `builtin_scraper.py`. Japan routing is handled in
`job_matcher_mcp.py` using `DEFAULT_JAPAN_LOCATIONS`.

### 3. Update `scrape_jobs()` routing logic

Find the block in `scrape_jobs()` that splits `us_locations` and `eu_locations`:

```python
# CURRENT CODE (around line 95-110):
us_locations = [l for l in locations if any(l.lower() in x.lower() for x in builtin_locs)]
eu_locations = [l for l in locations if l not in us_locations]

dfs = []
if us_locations:
    df_us = await scrape_builtin(...)
    if not df_us.empty:
        dfs.append(df_us)
if eu_locations:
    df_eu = await scrape_serper(...)
    if not df_eu.empty:
        dfs.append(df_eu)
```

Replace with:

```python
# UPDATED CODE — three-way routing:
us_locations = [l for l in locations if any(l.lower() in x.lower() for x in builtin_locs)]
jp_locations = [l for l in locations if any(l.lower() in j.lower() for j in DEFAULT_JAPAN_LOCATIONS)]
eu_locations = [l for l in locations if l not in us_locations and l not in jp_locations]

dfs = []
if us_locations:
    df_us = await scrape_builtin(keywords, us_locations, max_results_per_query, job_domain=job_domain, source_map=source_map)
    if not df_us.empty:
        dfs.append(df_us)
if eu_locations:
    df_eu = await scrape_serper(keywords, eu_locations, max_results_per_query, job_domain=job_domain, ats_domains=ats_domains)
    if not df_eu.empty:
        dfs.append(df_eu)
if jp_locations:
    df_jp = await scrape_japan(keywords, jp_locations, max_results_per_query, job_domain=job_domain)
    if not df_jp.empty:
        dfs.append(df_jp)
```

### 4. Update `scrape_jobs()` docstring

Add to the Parameters section:

```python
"""
...
- locations: ...
  Special values that trigger Japanese board routing (via japan_scraper.py):
  "Japan", "Tokyo", "Japanese companies", "Japan remote"
  These locations route to daijob.com, gaijinpot.com, careercross.com, jp.japanese-jobs.com
  instead of BuiltIn or Serper. Location string is NOT appended to queries for these boards.

Example — Japanese companies seeking EMEA channel reps:
    scrape_jobs(
        keywords=["EMEA sales", "Europe channel partner", "channel manager Europe"],
        locations=["Japan"],
        job_domain="sales",
        max_total_queries=6
    )
...
"""
```

---

## No Changes Required To

| File | Status | Reason |
|---|---|---|
| `serper_scraper.py` | ✅ Untouched | EU/global Serper logic unchanged |
| `builtin_scraper.py` | ✅ Untouched | US/BuiltIn logic unchanged |
| `extractor.py` | ✅ Untouched | LLM extraction reused as-is |
| `job_scorer.py` | ✅ Untouched | Embedding utilities unchanged |
| `.env` | ✅ Untouched | No new env vars needed |
| `pyproject.toml` | ✅ Untouched | No new dependencies (reuses aiohttp, beautifulsoup4, requests) |

---

## Testing After Implementation

### Step 1 — Unit test the query builder

```python
# tests/test_japan_scraper.py

from japan_scraper import build_japan_queries, DEFAULT_JAPAN_BOARDS

def test_no_location_in_queries():
    """Japan queries must NOT contain location strings."""
    queries = build_japan_queries(
        keywords=["EMEA sales", "Europe channel partner"],
        boards=DEFAULT_JAPAN_BOARDS,
    )
    for q in queries:
        assert "Tokyo" not in q
        assert "Japan" not in q
        assert "Berlin" not in q

def test_query_contains_keyword():
    queries = build_japan_queries(["EMEA sales"], DEFAULT_JAPAN_BOARDS)
    assert any("EMEA sales" in q for q in queries)

def test_query_contains_all_boards():
    queries = build_japan_queries(["EMEA sales"], DEFAULT_JAPAN_BOARDS)
    for board in DEFAULT_JAPAN_BOARDS:
        assert any(board in q for q in queries)

def test_one_query_per_keyword():
    queries = build_japan_queries(["kw1", "kw2", "kw3"], DEFAULT_JAPAN_BOARDS)
    assert len(queries) == 3   # one per keyword, not keyword × location
```

### Step 2 — Live integration test

After Claude Desktop restart:

```python
# In Claude chat:
scrape_jobs(
    keywords=["EMEA sales", "Europe channel partner", "channel manager Europe"],
    locations=["Japan"],
    job_domain="sales",
    max_total_queries=6
)
```

Expected result:
- `queries_run: 3` (one per keyword, no location multiplication)
- `source` column in CSV shows `"japan_boards"`
- Job titles in CSV reference European/EMEA scope roles
- `company` and `location` fields populated by LLM extraction

### Step 3 — Routing test (all three scrapers in one call)

```python
scrape_jobs(
    keywords=["channel manager"],
    locations=["New York", "London", "Japan"],
    job_domain="sales",
    max_total_queries=6
)
```

Expected:
- New York → `builtin_scraper` (source: `builtin`)
- London → `serper_scraper` (source: `serper`)
- Japan → `japan_scraper` (source: `japan_boards`)
- All three merge into single CSV

---

## Expected Japanese Job Board Results

Based on manual searches, these keyword patterns return results on Japanese boards:

| Keyword | Likely result type |
|---|---|
| `"EMEA sales"` | Japanese tech/manufacturing companies hiring EU reps |
| `"Europe channel partner"` | Vendors seeking EU distribution partners |
| `"channel manager Europe"` | Partner/channel roles covering EU from Japan base |
| `"European market"` | Expansion roles — BD, sales, country manager |
| `"global sales" "Europe"` | Cross-border AE roles at Japanese tech firms |

**Industries most likely to have results:**
- Industrial technology (Fanuc, Keyence, Omron, Yokogawa)
- Enterprise software (Fujitsu, NEC, Hitachi)
- Cybersecurity (Trend Micro, NTT Security)
- Semiconductor / electronics (Renesas, Murata)
- Robotics / automation (FANUC, Yaskawa)

**Realistic expectations:**
- Volume will be lower than EU/US boards (10–30 results vs 100–300)
- Quality may be higher — these are active expansion roles, not spray postings
- Some results may be in Japanese — LLM extractor handles this (translate to English)
- `company` and `location` fields may be NaN more often — thin HTML on some boards

---

## Suggested Keywords for First Run

```python
scrape_jobs(
    keywords=[
        "EMEA sales",
        "Europe channel partner",
        "channel manager Europe",
        "European market expansion",
        "global sales Europe"
    ],
    locations=["Japan"],
    job_domain="sales",
    max_total_queries=6,   # 5 keywords × 1 "location" = 5 queries, within cap
)
```

Then score and rank against Denis's v0.4 profile as usual.
