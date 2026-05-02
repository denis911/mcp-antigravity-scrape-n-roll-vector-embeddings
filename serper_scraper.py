import os
import asyncio
import logging
import requests
import aiohttp
from bs4 import BeautifulSoup

SERPER_URL = "https://google.serper.dev/search"

# Common ATS and job boards to restrict the search and get cleaner data
ATS_DOMAINS = [
    "boards.greenhouse.io",
    "jobs.lever.co",
    "apply.workable.com",
    "jobs.ashbyhq.com",
]

def build_serper_queries(keywords: list[str], locations: list[str]) -> list[str]:
    """Build Serper search queries using ATS footprints."""
    domains_str = " OR ".join([f"site:{d}" for d in ATS_DOMAINS])
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

async def get_raw_jobs_serper(keywords: list[str], locations: list[str]) -> list[dict]:
    """High-level function to perform search and fetch phases."""
    queries = build_serper_queries(keywords, locations)
    
    logging.info(f"Generated {len(queries)} Serper queries based on ATS domains.")
    organic_results = await search_jobs_serper(queries)
    
    logging.info(f"Found {len(organic_results)} unique organic job links. Fetching descriptions...")
    enriched_results = await fetch_all_descriptions(organic_results)
    
    return enriched_results
