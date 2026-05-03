import os
import asyncio
import logging
import requests
import aiohttp
from bs4 import BeautifulSoup
import pandas as pd

from extractor import extract_structured_data_async, detect_domain, ensure_canonical_columns

SERPER_URL = "https://google.serper.dev/search"

# Common ATS and job boards to restrict the search and get cleaner data
DEFAULT_ATS_DOMAINS = [
    # --- Applicant Tracking Systems (ATS) ---
    "boards.greenhouse.io",
    "jobs.lever.co",
    "apply.workable.com",
    "jobs.ashbyhq.com",
    
    # --- Job Boards & Startup Platforms ---
    # Wellfound (formerly AngelList): Lists thousands of tech and startup jobs across Europe, including sales roles at funded companies.
    "wellfound.com",
    
    # Y Combinator Jobs: Focuses on sales positions at YC-funded startups in Europe, with direct founder connections.
    "ycombinator.com/jobs",
    
    # The Hub: Specializes in Nordic startup jobs, including sales, across Europe and remote options.
    "thehub.io",
    
    # TopStartups.io: Aggregates startup sales and other roles in Europe, updated daily.
    "topstartups.io",
    
    # Welcome to the Jungle: Europe's leading board for modern jobs, covering France, Germany, Spain, and more, with sales listings; noted as scrapable.
    "welcometothejungle.com",
    
    # EURES: Official EU portal with millions of jobs across Europe, filterable by sales/startups; has public scraping actors confirming accessibility
    "eures.europa.eu",
]

def build_serper_queries(keywords: list[str], locations: list[str], ats_domains: list[str]) -> list[str]:
    """Build Serper search queries using ATS footprints."""
    domains_str = " OR ".join([f"site:{d}" for d in ats_domains])
    ats_footprint = f"({domains_str})"
    
    queries = []
    for kw in keywords:
        for loc in locations:
            queries.append(f'{ats_footprint} "{kw}" "{loc}"')
    return queries

def _serper_search_sync(query: str) -> list[dict]:
    """Sync function to call Serper API."""
    api_key = os.environ.get("SERPER_API_KEY")
    if not api_key:
        raise ValueError("SERPER_API_KEY environment variable is not set. Please add it to your environment variables.")

    headers = {
        "X-API-KEY": api_key,
        "Content-Type": "application/json",
    }
    # num=20 fetches up to 20 organic results.
    payload = {"q": query, "num": 20}
    
    try:
        response = requests.post(SERPER_URL, headers=headers, json=payload, timeout=15)
        response.raise_for_status()
        return response.json().get("organic", [])
    except Exception as e:
        logging.error(f"Error querying Serper for '{query}': {e}")
        return []

async def search_jobs_serper(queries: list[str]) -> list[dict]:
    """Async wrapper to run Serper queries concurrently and deduplicate URLs."""
    loop = asyncio.get_event_loop()
    
    tasks = [
        loop.run_in_executor(None, _serper_search_sync, q)
        for q in queries
    ]
    
    results = await asyncio.gather(*tasks)
    
    # Flatten and deduplicate by URL
    seen_urls = set()
    deduped = []
    for batch in results:
        for item in batch:
            url = item.get("link")
            if url and url not in seen_urls:
                seen_urls.add(url)
                # Ensure structure resembles what scraper.py expects from Apify
                deduped.append({
                    "title": item.get("title", ""),
                    "url": url,
                    "company": "", # Company will be extracted by LLM
                    "description_snippet": item.get("snippet", ""),
                    "source": "serper"
                })
                
    return deduped

async def fetch_job_html(url: str, session: aiohttp.ClientSession) -> str:
    """Fetch the raw HTML from the job posting URL."""
    try:
        async with session.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}) as response:
            if response.status == 200:
                html = await response.text()
                soup = BeautifulSoup(html, "html.parser")
                # Remove script and style elements to clean text
                for script in soup(["script", "style", "noscript", "header", "footer", "nav"]):
                    script.extract()
                return soup.get_text(separator=" ", strip=True)
            else:
                logging.warning(f"Failed to fetch {url}: HTTP {response.status}")
                return ""
    except Exception as e:
        logging.warning(f"Error fetching {url}: {e}")
        return ""

async def fetch_all_descriptions(organic_results: list[dict]) -> list[dict]:
    """Concurrently fetch job descriptions for all organic results."""
    enriched_results = []
    
    connector = aiohttp.TCPConnector(limit=50)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = []
        for item in organic_results:
            tasks.append(fetch_job_html(item["url"], session))
            
        html_texts = await asyncio.gather(*tasks)
        
        for item, html_text in zip(organic_results, html_texts):
            # Use full HTML text if fetch succeeded, else fallback to snippet
            item["description_text"] = html_text if len(html_text) > 100 else item["description_snippet"]
            item["description"] = item["description_text"] # Map for LLM extraction
            enriched_results.append(item)
            
    return enriched_results

async def get_raw_jobs_serper(keywords: list[str], locations: list[str], ats_domains: list[str]) -> list[dict]:
    """High-level function to perform search and fetch phases."""
    queries = build_serper_queries(keywords, locations, ats_domains)
    
    logging.info(f"Generated {len(queries)} Serper queries based on ATS domains.")
    organic_results = await search_jobs_serper(queries)
    
    logging.info(f"Found {len(organic_results)} unique organic job links. Fetching descriptions...")
    enriched_results = await fetch_all_descriptions(organic_results)
    
    return enriched_results

async def scrape_serper(
    keywords: list[str],
    locations: list[str],
    max_per_query: int = 100,  # Included for consistent interface though Serper is limited to its own paging
    job_domain: str | None = None,
    ats_domains: list[str] | None = None
) -> pd.DataFrame:
    """
    Returns a DataFrame with the canonical schema columns defined in TASK.md
    scraped via Serper.
    """
    job_domain = job_domain or detect_domain(keywords)
    ats_domains = ats_domains or DEFAULT_ATS_DOMAINS
    logging.info(f"Using job_domain: {job_domain}")
    
    logging.info(f"Starting parallel Serper scrape...")
    raw_results = await get_raw_jobs_serper(keywords, locations, ats_domains)
    logging.info(f"Serper collected {len(raw_results)} unique items. Proceeding to LLM extraction...")
    
    if raw_results:
        structured_results = await extract_structured_data_async(raw_results, domain=job_domain)
    else:
        structured_results = []
        
    df = pd.DataFrame(structured_results)
    return ensure_canonical_columns(df)
