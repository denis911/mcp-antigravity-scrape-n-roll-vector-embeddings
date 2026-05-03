import os
import logging
import asyncio
import pandas as pd
from apify_client import ApifyClient

from extractor import extract_structured_data_async, detect_domain, ensure_canonical_columns

# Default location mappings
DEFAULT_SOURCE_MAP = {
    # BuiltIn — US cities
    "builtin": [
        "New York", "San Francisco", "Boston", "Chicago", "Austin",
        "Seattle", "Denver", "Los Angeles", "Remote", "US remote", "USA"
    ],
    # LinkedIn — global, requires different Apify actor
    "linkedin": [
        "London", "Berlin", "Amsterdam", "Paris", "Prague", "Vienna",
        "Zurich", "Munich", "Barcelona", "Stockholm", "Copenhagen",
        "Warsaw", "Remote Europe", "EU remote"
    ],
}

APIFY_ACTORS = {
    "builtin": {
        "primary": "shahidirfan/builtin-jobs-scraper",
        "fallback": "IhQuCmT40q1tetuv3",
    },
    "linkedin": {
        "primary": "nikhuge/advanced-linkedin-jobs-scraper-with-ai",  
        "fallback": "scrapier/linkedin-search-jobs-scraper",
    },
}

def _scrape_single_url(client: ApifyClient, url: str, actors: dict, max_items: int) -> list[dict]:
    """Blocking call to scrape a single URL, intended for run_in_executor."""
    logging.info(f"Scraping URL: {url}")
    run_input = {
        "startUrl": url,
        "results_wanted": max_items,
        "max_pages": 100
    }
    try:
        try:
            logging.info(f"Calling primary actor: {actors['primary']}")
            run = client.actor(actors['primary']).call(run_input=run_input)
        except Exception as e:
            error_msg = str(e).lower()
            if any(k in error_msg for k in ["subscription", "payment", "paid", "403"]):
                logging.warning(f"Primary actor restricted. Falling back to {actors['fallback']}")
                run = client.actor(actors['fallback']).call(run_input=run_input)
            else:
                raise

        dataset_id = run["defaultDatasetId"]
        items = list(client.dataset(dataset_id).iterate_items())
        logging.info(f"Got {len(items)} items from {url}")
        return items
    except Exception as e:
        logging.error(f"Error scraping {url}: {e}")
        return []

async def scrape_apify(urls_by_source: dict[str, list[str]], max_items: int) -> list[dict]:
    token = os.environ.get("APIFY_API_TOKEN")
    if not token:
        raise ValueError("APIFY_API_TOKEN environment variable is not set.")
        
    client = ApifyClient(token)
    loop = asyncio.get_event_loop()
    
    tasks = []
    for source, urls in urls_by_source.items():
        actors = APIFY_ACTORS.get(source, APIFY_ACTORS["builtin"])
        for url in urls:
            tasks.append(loop.run_in_executor(None, _scrape_single_url, client, url, actors, max_items))
            
    if not tasks:
        return []
        
    results = await asyncio.gather(*tasks)
    
    all_results = []
    for r in results:
        all_results.extend(r)
            
    # Dedup by URL
    seen_urls = set()
    deduped = []
    for item in all_results:
        item_url = item.get("url")
        if item_url and item_url not in seen_urls:
            seen_urls.add(item_url)
            deduped.append(item)
            
    # Normalize missing fields mapped by different actors
    for item in deduped:
        if not item.get("workplace_type") and item.get("workType"):
            item["workplace_type"] = item["workType"]
            
        if not item.get("description") and item.get("description_text"):
            item["description"] = item["description_text"]
        if not item.get("postedDate") and item.get("date_posted"):
            item["postedDate"] = item["date_posted"]
            
        if not item.get("salary") and item.get("salary_json_min"):
            min_val = item.get("salary_json_min")
            max_val = item.get("salary_json_max")
            currency = item.get("salary_json_currency", "USD")
            if min_val and max_val:
                item["salary"] = f"{currency} {min_val:,} - {max_val:,}"
            elif min_val:
                item["salary"] = f"{currency} {min_val:,}+"

        if not item.get("experienceLevel") and item.get("seniority"):
            item["experienceLevel"] = item["seniority"]

        if not item.get("category_raw") and item.get("category"):
            item["category_raw"] = item["category"]

    return deduped

def build_urls(keywords: list[str], locations: list[str], source_map: dict | None = None) -> dict[str, list[str]]:
    """Returns {source_name: [url1, url2, ...]} grouped by scraping source."""
    source_map = source_map or DEFAULT_SOURCE_MAP
    urls_by_source = {"builtin": [], "linkedin": []}

    for kw in keywords:
        for loc in locations:
            kw_url = kw.replace(" ", "+")
            loc_url = loc.replace(" ", "+")

            # Determine source by location
            if any(loc.lower() in l.lower() for l in source_map.get("linkedin", [])):
                url = f"https://www.linkedin.com/jobs/search/?keywords={kw_url}&location={loc_url}"
                urls_by_source["linkedin"].append(url)
            else:
                url = f"https://builtin.com/jobs?search={kw_url}&location={loc_url}"
                urls_by_source["builtin"].append(url)

    return urls_by_source

async def scrape_builtin(
    keywords: list[str],
    locations: list[str],
    max_per_query: int = 100,
    job_domain: str | None = None,
    source_map: dict | None = None,
) -> pd.DataFrame:
    """
    Returns a DataFrame with the canonical schema columns defined in TASK.md
    scraped via Apify.
    """
    job_domain = job_domain or detect_domain(keywords)
    logging.info(f"Using job_domain: {job_domain}")
    
    urls_by_source = build_urls(keywords, locations, source_map)
    
    logging.info(f"Starting parallel Apify scrape...")
    raw_results = await scrape_apify(urls_by_source, max_per_query)
    logging.info(f"Apify collected {len(raw_results)} unique items. Proceeding to LLM extraction...")
    
    if raw_results:
        structured_results = await extract_structured_data_async(raw_results, domain=job_domain)
    else:
        structured_results = []
        
    df = pd.DataFrame(structured_results)
    return ensure_canonical_columns(df)
