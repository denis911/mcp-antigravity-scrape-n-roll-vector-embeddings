"""
linkedin_scraper.py — LinkedIn Job Scraper via Apify
=====================================================
Scrapes LinkedIn jobs for EU/global locations using Apify actors.

Primary actor:  shahidirfan~LinkedIn-Job-Scraper  (pay per use, lightweight)
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
        "primary": "shahidirfan~LinkedIn-Job-Scraper",
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

# Fix (matches OpenAPI schema exactly):
def _build_primary_input(kw: str, loc: str, max_items: int) -> dict:
    return {
        "query": kw,          # ← correct field name
        "location": loc,
        "maxJobs": max_items,
        "timeRange": "30d",   # matches enum: "anytime"|"24h"|"7d"|"30d"
        "collectOnly": False,
        "proxyConfiguration": {"useApifyProxy": True},
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
