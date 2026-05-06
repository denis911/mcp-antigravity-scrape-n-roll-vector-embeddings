# EXPAND-EU.md — Global Job Scraping via Serper API

## Objective

The current `scraper.py` implementation relies on BuiltIn (which is US-centric) and LinkedIn Apify actors (which can be flaky or blocked). To expand the pipeline to find job postings globally (especially in Europe and other non-US markets), we will integrate the **Serper API** (Google Search API).

This plan details how to replace or augment the existing pipeline to use Serper API, targeting Applicant Tracking Systems (ATS) directly to find clean, legitimate job postings worldwide.

> [!NOTE]
> **Why Serper + ATS Footprints?**
> Standard Google searches for jobs often return SEO-spam job aggregator sites. By explicitly querying ATS platforms (Greenhouse, Lever, Workable, etc.), we get direct links to the primary source of the job posting, leading to cleaner data and easier HTML parsing.

## Approach: Two-Step Search and Extract

Unlike Apify which returns structured job records with full descriptions, Serper returns Google organic search results (title, link, and a short 160-character snippet). Because the snippet is too short for the LLM to accurately extract the `tech_stack`, `salary`, and `seniority`, the new pipeline requires an intermediate fetching step:

1. **Search Phase**: Query Serper with ATS "dorks" (e.g., `site:jobs.lever.co "Data Scientist" "Berlin"`).
2. **Fetch Phase**: Download the raw HTML from the organic links concurrently and extract the text.
3. **Structure Phase**: Pass the extracted text through the existing `extract_structured_data_async()` OpenAI pipeline.

---

## User Review Required

> [!WARNING]
> **Dependency Addition**
> The current project uses `requests`. Fetching 50+ job description URLs synchronously with `requests` will be very slow. I propose adding `aiohttp` to `pyproject.toml` for efficient, concurrent fetching of job descriptions. Let me know if you prefer to stick to `requests` wrapped in threads instead.

> [!IMPORTANT]
> **Scraping Strategy**
> Scraping random job board URLs can yield messy HTML. Focusing on known ATS platforms like `greenhouse.io`, `lever.co`, `workable.com`, and `ashbyhq.com` using Google `site:` operators will provide consistent and clean HTML. Is this ATS-focused strategy acceptable, or do you want to search the entire web and attempt to parse any URL returned?

---

## Proposed Changes

### 1. Configuration (`.env`)
- Add `SERPER_API_KEY` to the expected environment variables.
- Optional: Add `SCRAPER_BACKEND=serper` to toggle between Apify and Serper if we want to keep both.

### 2. URL/Query Builder
Instead of building Apify URLs, we will build Google Search queries utilizing ATS footprints.

```python
def build_serper_queries(keywords: list[str], locations: list[str]) -> list[str]:
    # ATS footprints to filter out aggregator spam
    ats_footprint = "(site:boards.greenhouse.io OR site:jobs.lever.co OR site:apply.workable.com OR site:jobs.ashbyhq.com)"
    queries = []
    for kw in keywords:
        for loc in locations:
            queries.append(f'{ats_footprint} "{kw}" "{loc}"')
    return queries
```

### 3. Implement Serper Search
We will adapt your `_serper_search` snippet into an async-compatible wrapper.

```python
import os
import requests
import asyncio

SERPER_URL = "https://google.serper.dev/search"

def _serper_search(query: str) -> list[dict]:
    api_key = os.environ.get("SERPER_API_KEY")
    if not api_key:
        raise ValueError("SERPER_API_KEY environment variable is not set.")

    headers = {
        "X-API-KEY": api_key,
        "Content-Type": "application/json",
    }
    payload = {"q": query, "num": 20} # Retrieve top 20 organic results per query
    
    response = requests.post(SERPER_URL, headers=headers, json=payload, timeout=15)
    response.raise_for_status()
    return response.json().get("organic", [])
    
async def search_jobs_serper(queries: list[str]) -> list[dict]:
    # Run _serper_search in an executor for each query concurrently
    # Deduplicate results by URL
```

### 4. Async HTML Fetcher
We will fetch the actual job description text from the URLs returned by Serper.

```python
import aiohttp
from bs4 import BeautifulSoup

async def fetch_job_html(url: str, session: aiohttp.ClientSession) -> str:
    try:
        async with session.get(url, timeout=10) as response:
            if response.status == 200:
                html = await response.text()
                # Use BeautifulSoup to extract text from the body
                soup = BeautifulSoup(html, "html.parser")
                return soup.get_text(separator=" ", strip=True)
    except Exception as e:
        return ""
        
async def fetch_all_descriptions(organic_results: list[dict]) -> list[dict]:
    # Fetch all URLs concurrently and append 'description_text' to the result dictionary
```

### 5. Integration into `scraper.py`
Update the main `scrape()` function to orchestrate the new flow:

#### [MODIFY] `scraper.py`
- Add `build_serper_queries`.
- Add `_serper_search` and `fetch_all_descriptions`.
- Update `scrape()` to route traffic through the Serper pipeline instead of Apify.
- Ensure the resulting Dataframe matches the canonical schema (filling Apify-specific fields like `salary_json_min` with empty strings, relying entirely on the LLM's `extract_structured_data_async()` output for data normalization).

#### [MODIFY] `pyproject.toml`
- Add `aiohttp>=3.9.0` to dependencies.

---

## Verification Plan

### Automated Tests
- Run `uv run pytest tests/ -v` to ensure existing embedding and ranking tests pass.
- Write a new test in `test_ranking.py` to mock the Serper API response and ensure the `scrape()` function builds the expected Dataframe schema.

### Manual Verification
- Start the server using `uv run job_matcher_mcp.py`.
- Connect via MCP Inspector or Claude Desktop.
- Run `scrape_jobs(keywords=["Machine Learning Engineer"], locations=["Berlin", "Amsterdam"])`.
- Verify that `list_saved_csvs()` shows the new scrape file, and that inspecting the CSV reveals actual European job postings with populated `tech_stack` and `salary` columns.
