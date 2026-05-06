# EXPAND-LI.md — LinkedIn Job Scraper

## Goal

Implement `linkedin_scraper.py` — the third Apify-based scraper, targeting LinkedIn jobs
for EU locations. The placeholder file exists but is empty. This spec tells the coding
agent exactly what to build and how it differs from `builtin_scraper.py`.

---

## Context — Why LinkedIn Separate from BuiltIn

`builtin_scraper.py` already has LinkedIn actor config (`APIFY_ACTORS["linkedin"]`) and
URL routing (`DEFAULT_SOURCE_MAP["linkedin"]`) but the actual LinkedIn scraping is
delegated to `linkedin_scraper.py` which is currently empty.

The separation exists because:

1. **Different Apify actor** — LinkedIn requires a dedicated actor with LinkedIn-specific
   input schema (not `startUrl` + `results_wanted` like BuiltIn)
2. **Different URL format** — `linkedin.com/jobs/search/?keywords=...&location=...`
   vs `builtin.com/jobs?search=...&location=...`
3. **Different output fields** — LinkedIn returns `jobUrl`, `companyName`, `location`,
   `descriptionHtml`, `employmentType`, `seniorityLevel` vs BuiltIn's schema
4. **Anti-bot considerations** — LinkedIn requires residential proxies + small batches
   + concurrency limits. BuiltIn is more forgiving.
5. **Free tier strategy** — Keep LinkedIn runs small (10–30 jobs max) to preserve
   Apify credits. BuiltIn can run larger batches.

---

## Apify Actors

```python
APIFY_ACTORS = {
    "linkedin": {
        "primary": "shahidirfan/linkedin-job-scraper",
        "fallback": "scrapier/linkedin-search-jobs-scraper",
    },
}
```

### Primary: `shahidirfan/linkedin-job-scraper`
- Pay per usage (no fixed monthly fee)
- Lightweight — focuses on core fields
- Max ~1,000 jobs/run
- Supports keywords, location, time range, proxies
- Rating: 5.0 (limited reviews but recent and optimised)
- **Free tier strategy:** test with `maxJobs=20`, datacenter proxies first

### Fallback: `scrapier/linkedin-search-jobs-scraper`
- Activated if primary fails with subscription/payment/403 error
- Same field structure, different implementation

### Actor input schema (shahidirfan/linkedin-job-scraper)

```python
run_input = {
    "keyword": kw,           # job title / keyword string (NOT URL)
    "location": loc,         # city name string
    "maxJobs": max_items,    # cap per query — keep small (20-30 for free tier)
    "timeRange": "past_week",  # "any_time" | "past_month" | "past_week" | "past_24_hours"
    # Optional: proxy config — omit for datacenter (cheaper), add residential if blocked
    # "proxyConfig": {"useApifyProxy": True, "apifyProxyGroups": ["RESIDENTIAL"]}
}
```

### Fallback actor input schema (scrapier/linkedin-search-jobs-scraper)

```python
run_input = {
    "keyword": kw,
    "location": loc,
    "count": max_items,
}
```

---

## Field Mapping — LinkedIn → Canonical Schema

LinkedIn actors return different field names than BuiltIn. Map them during normalisation.

| LinkedIn field | Canonical field | Notes |
|---|---|---|
| `jobUrl` | `url` | Primary URL field |
| `link` | `url` | Fallback if `jobUrl` empty |
| `companyName` | `company` | |
| `title` | `title` | Usually clean |
| `location` | `location` | May include "Remote" |
| `descriptionHtml` | `description_html` | Full HTML |
| `description` | `description_text` | Plain text version |
| `descriptionText` | `description_text` | Alternative field name |
| `employmentType` | `employment_type` | "Full-time", "Contract", etc. |
| `seniorityLevel` | `seniority` | "Mid-Senior level", "Director", etc. |
| `postedAt` | `date_posted` | ISO date or relative string |
| `postedDate` | `date_posted` | Alternative |
| `contractType` | `employment_type` | Fallback |
| `workplaceType` | `workplace_type` | "Remote", "Hybrid", "On-site" |
| `salary` | `salary` | May be empty — LinkedIn hides salaries often |

**Seniority normalisation** — LinkedIn uses verbose strings, map to our canonical values:

```python
SENIORITY_MAP = {
    "internship": "Intern",
    "entry level": "Junior",
    "associate": "Mid",
    "mid-senior level": "Senior",
    "director": "Director",
    "executive": "VP",
    "not applicable": None,
}
```

---

## Complete `linkedin_scraper.py` Implementation

Replace the placeholder with this full implementation:

```python
"""
linkedin_scraper.py — LinkedIn Job Scraper via Apify
=====================================================
Scrapes LinkedIn jobs for EU/global locations using Apify actors.

Primary actor:  shahidirfan/linkedin-job-scraper  (pay per use, lightweight)
Fallback actor: scrapier/linkedin-search-jobs-scraper

Key differences from builtin_scraper.py:
  - Actor input uses keyword+location strings (not startUrls)
  - Different output field names — mapped to canonical schema
  - Smaller max_items per query (free tier conservation)
  - Seniority values need normalisation ("Mid-Senior level" → "Senior")
  - Anti-bot: small batches, optional residential proxies

Usage (called from job_matcher_mcp.py):
    df = await scrape_linkedin(
        keywords=["channel manager", "partner manager"],
        locations=["London", "Berlin", "Amsterdam"],
        max_per_query=20,
        job_domain="sales"
    )
"""

import os
import logging
import asyncio
import pandas as pd
from apify_client import ApifyClient

from extractor import extract_structured_data_async, detect_domain, ensure_canonical_columns

# ── Actor Config ──────────────────────────────────────────────────────────────

APIFY_ACTORS = {
    "linkedin": {
        "primary": "shahidirfan/linkedin-job-scraper",
        "fallback": "scrapier/linkedin-search-jobs-scraper",
    },
}

# LinkedIn seniority strings → canonical values
SENIORITY_MAP = {
    "internship": "Intern",
    "entry level": "Junior",
    "associate": "Mid",
    "mid-senior level": "Senior",
    "senior": "Senior",
    "director": "Director",
    "executive": "VP",
    "not applicable": None,
    "": None,
}

# ── Actor Input Builders ──────────────────────────────────────────────────────

def _build_primary_input(kw: str, loc: str, max_items: int) -> dict:
    """Input schema for shahidirfan/linkedin-job-scraper."""
    return {
        "keyword": kw,
        "location": loc,
        "maxJobs": max_items,
        "timeRange": "past_month",
        # Uncomment to use residential proxies (more credits, more reliable):
        # "proxyConfig": {
        #     "useApifyProxy": True,
        #     "apifyProxyGroups": ["RESIDENTIAL"]
        # }
    }

def _build_fallback_input(kw: str, loc: str, max_items: int) -> dict:
    """Input schema for scrapier/linkedin-search-jobs-scraper."""
    return {
        "keyword": kw,
        "location": loc,
        "count": max_items,
    }

# ── Field Normalisation ───────────────────────────────────────────────────────

def _normalise_seniority(raw: str | None) -> str | None:
    """Map LinkedIn verbose seniority strings to canonical values."""
    if not raw:
        return None
    return SENIORITY_MAP.get(raw.lower().strip(), raw)

def _normalise_item(item: dict) -> dict:
    """
    Map LinkedIn-specific field names to canonical schema.
    Called on each raw item before LLM extraction.
    """
    # URL — try multiple field names
    if not item.get("url"):
        item["url"] = item.get("jobUrl") or item.get("link") or ""

    # Company
    if not item.get("company"):
        item["company"] = item.get("companyName") or ""

    # Description — try multiple field names
    if not item.get("description_text"):
        item["description_text"] = (
            item.get("descriptionText")
            or item.get("description")
            or item.get("descriptionHtml")  # fallback to HTML if no plain text
            or ""
        )
    if not item.get("description"):
        item["description"] = item["description_text"]

    if not item.get("description_html"):
        item["description_html"] = item.get("descriptionHtml") or ""

    # Employment type
    if not item.get("employment_type"):
        item["employment_type"] = (
            item.get("employmentType")
            or item.get("contractType")
            or ""
        )

    # Seniority — normalise verbose LinkedIn strings
    raw_seniority = item.get("seniorityLevel") or item.get("seniority") or ""
    item["seniority"] = _normalise_seniority(raw_seniority)

    # Date posted
    if not item.get("date_posted"):
        item["date_posted"] = item.get("postedAt") or item.get("postedDate") or ""

    # Workplace type
    if not item.get("workplace_type"):
        item["workplace_type"] = item.get("workplaceType") or item.get("workType") or ""

    # Source tag
    item["source"] = "linkedin"

    return item

# ── Single Query Scrape ───────────────────────────────────────────────────────

def _scrape_single_query(
    client: ApifyClient,
    kw: str,
    loc: str,
    max_items: int,
) -> list[dict]:
    """
    Blocking call to scrape a single keyword+location pair.
    Designed to run in asyncio executor.
    Tries primary actor, falls back on subscription/payment/403 errors.
    """
    actors = APIFY_ACTORS["linkedin"]
    logging.info(f"LinkedIn scrape: '{kw}' in '{loc}' (max {max_items})")

    def _call_actor(actor_id: str, run_input: dict) -> list[dict]:
        run = client.actor(actor_id).call(run_input=run_input)
        dataset_id = run["defaultDatasetId"]
        items = list(client.dataset(dataset_id).iterate_items())
        logging.info(f"LinkedIn: {len(items)} items from '{kw}'/'{loc}' via {actor_id}")
        return items

    try:
        try:
            run_input = _build_primary_input(kw, loc, max_items)
            return _call_actor(actors["primary"], run_input)
        except Exception as e:
            error_msg = str(e).lower()
            if any(k in error_msg for k in ["subscription", "payment", "paid", "403"]):
                logging.warning(
                    f"LinkedIn primary actor restricted for '{kw}'/'{loc}'. "
                    f"Falling back to {actors['fallback']}"
                )
                run_input = _build_fallback_input(kw, loc, max_items)
                return _call_actor(actors["fallback"], run_input)
            raise
    except Exception as e:
        logging.error(f"LinkedIn scrape error for '{kw}'/'{loc}': {e}")
        return []

# ── Parallel Scraping ─────────────────────────────────────────────────────────

async def scrape_apify_linkedin(
    keywords: list[str],
    locations: list[str],
    max_items: int,
) -> list[dict]:
    """
    Run all keyword × location queries in parallel using asyncio executor.
    Deduplicates results by URL.
    """
    token = os.environ.get("APIFY_API_TOKEN")
    if not token:
        raise ValueError(
            "APIFY_API_TOKEN environment variable is not set. "
            "Add it to claude_desktop_config.json env block."
        )

    client = ApifyClient(token)
    loop = asyncio.get_event_loop()

    # Build tasks — one per keyword × location combination
    tasks = [
        loop.run_in_executor(None, _scrape_single_query, client, kw, loc, max_items)
        for kw in keywords
        for loc in locations
    ]

    if not tasks:
        return []

    results = await asyncio.gather(*tasks)

    # Flatten and deduplicate by URL
    seen_urls = set()
    deduped = []
    for batch in results:
        for item in batch:
            # Normalise fields before dedup check
            item = _normalise_item(item)
            url = item.get("url")
            if url and url not in seen_urls:
                seen_urls.add(url)
                deduped.append(item)
            elif not url:
                # Keep items without URL — LLM may still extract useful data
                deduped.append(item)

    logging.info(
        f"LinkedIn scraper: {len(deduped)} unique items from "
        f"{len(keywords) * len(locations)} queries"
    )
    return deduped

# ── Main Entry Point ──────────────────────────────────────────────────────────

async def scrape_linkedin(
    keywords: list[str],
    locations: list[str],
    max_per_query: int = 20,      # Keep LOW for free tier — LinkedIn is expensive
    job_domain: str | None = None,
) -> pd.DataFrame:
    """
    Scrape LinkedIn jobs for given keywords and EU/global locations.

    Interface matches scrape_builtin() and scrape_serper() for clean
    integration with job_matcher_mcp.py.

    Parameters:
        keywords      : Job title / skill keywords e.g. ["channel manager", "partner manager"]
        locations     : City names e.g. ["London", "Berlin", "Amsterdam"]
        max_per_query : Max jobs per keyword×location pair. Keep 10–30 for free tier.
                        Residential proxies needed for higher volumes.
        job_domain    : "sales", "gtm", "data", "biotech", "any" (auto-detected if None)

    Returns:
        pd.DataFrame with canonical schema columns (see TASK.md)

    Free tier guidance:
        $5 Apify credit/month.
        ~$0.01–0.05 per LinkedIn job scraped (datacenter proxies).
        With max_per_query=20 and 4 queries: ~80 jobs = ~$0.80–4.00 per run.
        Test with max_per_query=10 first to calibrate credit consumption.
    """
    job_domain = job_domain or detect_domain(keywords)
    logging.info(
        f"LinkedIn scraper: keywords={keywords}, locations={locations}, "
        f"max_per_query={max_per_query}, domain={job_domain}"
    )

    # Stage 1: Scrape
    raw_results = await scrape_apify_linkedin(keywords, locations, max_per_query)

    if not raw_results:
        logging.info("LinkedIn scraper: no results returned")
        return ensure_canonical_columns(pd.DataFrame())

    # Stage 2: LLM extraction (same extractor.py as all other scrapers)
    logging.info(f"LinkedIn scraper: running LLM extraction on {len(raw_results)} items...")
    structured_results = await extract_structured_data_async(raw_results, domain=job_domain)

    df = pd.DataFrame(structured_results)
    df = ensure_canonical_columns(df)

    logging.info(f"LinkedIn scraper: returning {len(df)} structured records")
    return df
```

---

## Changes to `job_matcher_mcp.py`

### 1. Update import

```python
# CURRENT:
from builtin_scraper import scrape_builtin, DEFAULT_SOURCE_MAP
from serper_scraper import scrape_serper
from japan_scraper import scrape_japan, DEFAULT_JAPAN_LOCATIONS

# ADD:
from linkedin_scraper import scrape_linkedin
```

### 2. Update routing logic in `scrape_jobs()`

Find the section where `eu_locations` routes to `scrape_serper`. LinkedIn locations
are currently in `DEFAULT_SOURCE_MAP["linkedin"]` in `builtin_scraper.py`. Add a
LinkedIn-specific routing tier between EU-Serper and US-BuiltIn:

```python
# Get location lists
s_map = source_map or DEFAULT_SOURCE_MAP
builtin_locs = s_map.get("builtin", [])
linkedin_locs = s_map.get("linkedin", [])   # NEW

us_locations = [l for l in locations if any(l.lower() in x.lower() for x in builtin_locs)]
jp_locations = [l for l in locations if any(l.lower() in j.lower() for j in DEFAULT_JAPAN_LOCATIONS)]
li_locations = [l for l in locations if any(l.lower() in x.lower() for x in linkedin_locs) and l not in jp_locations]   # NEW
eu_locations = [l for l in locations if l not in us_locations and l not in jp_locations and l not in li_locations]

dfs = []
if us_locations:
    df_us = await scrape_builtin(keywords, us_locations, max_results_per_query, job_domain=job_domain, source_map=source_map)
    if not df_us.empty:
        dfs.append(df_us)
if li_locations:                                                    # NEW BLOCK
    df_li = await scrape_linkedin(keywords, li_locations, max_results_per_query, job_domain=job_domain)
    if not df_li.empty:
        dfs.append(df_li)
if eu_locations:
    df_eu = await scrape_serper(keywords, eu_locations, max_results_per_query, job_domain=job_domain, ats_domains=ats_domains)
    if not df_eu.empty:
        dfs.append(df_eu)
if jp_locations:
    df_jp = await scrape_japan(keywords, jp_locations, max_results_per_query, job_domain=job_domain)
    if not df_jp.empty:
        dfs.append(df_jp)
```

### 3. Update `scrape_jobs()` docstring

Add to Parameters section:

```
- locations: EU city names (London, Berlin, Amsterdam, Paris, Prague, Vienna,
  Zurich, Munich, Barcelona, Stockholm, Copenhagen, Warsaw, Remote Europe,
  EU remote) automatically route to LinkedIn via Apify.
  IMPORTANT: LinkedIn scraping consumes more Apify credits than BuiltIn.
  Keep max_results_per_query at 10–20 for free tier ($5/month).
  Example: scrape_jobs(keywords=["channel manager"], locations=["London", "Berlin"])
```

---

## Routing Summary After Implementation

All four scrapers active, location-based routing:

| Location examples | Scraper | Actor |
|---|---|---|
| New York, San Francisco, Remote, USA | `builtin_scraper` | `shahidirfan/builtin-jobs-scraper` |
| London, Berlin, Amsterdam, Prague, EU remote | `linkedin_scraper` | `shahidirfan/linkedin-job-scraper` |
| Japan, Tokyo, Japanese companies | `japan_scraper` | Serper API |
| Any other location | `serper_scraper` | Serper API + ATS boards |

---

## No Changes Required To

| File | Status |
|---|---|
| `builtin_scraper.py` | ✅ Untouched — APIFY_ACTORS["linkedin"] entry can stay or be removed (no longer used by builtin) |
| `serper_scraper.py` | ✅ Untouched |
| `japan_scraper.py` | ✅ Untouched |
| `extractor.py` | ✅ Untouched — LLM extraction reused as-is |
| `job_scorer.py` | ✅ Untouched |
| `pyproject.toml` | ✅ Untouched — `apify-client` already in dependencies |
| `.env` | ✅ Untouched — `APIFY_API_TOKEN` already set |

Note: `builtin_scraper.py` has `APIFY_ACTORS["linkedin"]` with `nikhuge/advanced-linkedin-jobs-scraper-with-ai`
as primary. This entry is now superseded by `linkedin_scraper.py`. The coding agent
should remove or comment out the `"linkedin"` key from `APIFY_ACTORS` in
`builtin_scraper.py` to avoid confusion, but it will not cause errors if left as-is
since `scrape_builtin()` only uses `APIFY_ACTORS["builtin"]` for its own scraping.

---

## Free Tier Credit Management

LinkedIn is the most expensive scraper. Budget guidance:

| Scraper | Cost per job | 20 jobs/query | 4 queries/run |
|---|---|---|---|
| BuiltIn (Apify) | ~$0.002 | ~$0.04 | ~$0.16 |
| Serper (Google) | ~$0.001 | ~$0.02 | ~$0.08 |
| Japan boards | ~$0.001 | ~$0.02 | ~$0.06 |
| **LinkedIn (Apify)** | **~$0.01–0.05** | **~$0.20–1.00** | **~$0.80–4.00** |

**Recommendation for free tier ($5/month):**
- Run LinkedIn with `max_results_per_query=10–15`, not the default 100
- Test first run with `max_results_per_query=5` to calibrate actual credit consumption
- LinkedIn runs once per new keyword set, not daily
- BuiltIn can run daily; LinkedIn weekly

**Add to `.env`:**
```
LINKEDIN_MAX_PER_QUERY=15    # override default in scrape_linkedin() if needed
```

Optionally read this in `linkedin_scraper.py`:
```python
max_per_query = int(os.getenv("LINKEDIN_MAX_PER_QUERY", max_per_query))
```

---

## Testing After Implementation

### Step 1 — Unit test field normalisation

```python
# tests/test_linkedin_scraper.py

from linkedin_scraper import _normalise_item, _normalise_seniority

def test_seniority_normalisation():
    assert _normalise_seniority("Mid-Senior level") == "Senior"
    assert _normalise_seniority("Entry level") == "Junior"
    assert _normalise_seniority("Director") == "Director"
    assert _normalise_seniority("Internship") == "Intern"
    assert _normalise_seniority("Not Applicable") is None
    assert _normalise_seniority("") is None

def test_url_fallback():
    item = {"jobUrl": "https://linkedin.com/jobs/view/123"}
    result = _normalise_item(item)
    assert result["url"] == "https://linkedin.com/jobs/view/123"

def test_url_fallback_link():
    item = {"link": "https://linkedin.com/jobs/view/456"}
    result = _normalise_item(item)
    assert result["url"] == "https://linkedin.com/jobs/view/456"

def test_source_tag():
    item = {"title": "Channel Manager", "url": "https://linkedin.com/jobs/view/789"}
    result = _normalise_item(item)
    assert result["source"] == "linkedin"

def test_description_fallback_chain():
    item = {"descriptionText": "Job description here"}
    result = _normalise_item(item)
    assert result["description_text"] == "Job description here"
    assert result["description"] == "Job description here"
```

### Step 2 — Live integration test (small batch)

After Claude Desktop restart:

```python
scrape_jobs(
    keywords=["channel manager"],
    locations=["London"],
    max_results_per_query=5,    # SMALL — test credit consumption first
    job_domain="sales"
)
```

Expected:
- `jobs_scraped: 3–5`
- `source` column shows `"linkedin"`
- `url` fields contain `linkedin.com/jobs/` URLs
- `seniority` shows normalised values ("Senior" not "Mid-Senior level")
- Check Apify console for credit consumed before scaling up

### Step 3 — Full routing test

```python
scrape_jobs(
    keywords=["channel manager"],
    locations=["New York", "London", "Japan"],
    max_results_per_query=5,
    job_domain="sales"
)
```

Expected CSV `source` column breakdown:
- `builtin` rows (New York)
- `linkedin` rows (London)  
- `japan_boards` rows (Japan)

---

## Notes for Coding Agent

- `_scrape_single_query()` uses keyword+location strings directly as Apify actor input,
  NOT pre-built URLs. This is the key difference from `builtin_scraper.py`'s
  `_scrape_single_url()` which passes a full URL to Apify.
- The `_normalise_item()` function runs BEFORE LLM extraction (unlike BuiltIn where
  field mapping happens after). This ensures `description_text` is populated correctly
  for the LLM prompt.
- Do NOT use `run_input = {"startUrl": ..., "results_wanted": ...}` for LinkedIn actors
  — that is BuiltIn's input schema. LinkedIn actors want `{"keyword": ..., "location": ...}`.
- `max_per_query` defaults to 20 (not 100 like BuiltIn) to protect free tier credits.
- The fallback actor (`scrapier/linkedin-search-jobs-scraper`) uses `"count"` not
  `"maxJobs"` — this is handled by the separate `_build_fallback_input()` function.
