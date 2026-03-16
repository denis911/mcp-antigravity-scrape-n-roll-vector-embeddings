import os
import json
import logging
import time
import asyncio
import pandas as pd
from apify_client import ApifyClient
from openai import OpenAI

SOURCE_MAP = {
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
        # Standardize workplace_type early to avoid duplicate column issues
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

DOMAIN_PROMPTS = {
    "gtm": """
You are a GTM Engineering expert. Identify if the role is a Technical GTM role
(GTM Engineer, RevOps Engineer, Growth Engineer, Sales Engineer, Solutions Engineer).
Mark is_relevant=False for: pure Sales/Marketing, generic Software Engineering with no GTM tools.
""",
    "sales": """
You are an enterprise sales recruiter. Identify if the role is a B2B or enterprise sales role
(Account Executive, Sales Director, Channel Manager, Partner Manager, Solutions Consultant,
Business Development, Sales Engineer).
Mark is_relevant=False for: retail/consumer sales, inside sales SDR/BDR with no closing responsibility,
pure technical engineering with no customer-facing element.
""",
    "biotech": """
You are a computational biology recruiter. Identify if the role involves bioinformatics,
computational biology, genomics, ML in life sciences, or clinical data science
(Bioinformatics Engineer, Computational Biologist, ML Research Scientist - Life Sciences,
Data Scientist - Pharma, Genomics Engineer).
Mark is_relevant=False for: wet lab biology with no computational component, general software
engineering with no life sciences domain.
""",
    "data": """
You are a data engineering and ML recruiter. Identify if the role is a data/ML/AI engineering role
(Data Engineer, ML Engineer, Data Scientist, Analytics Engineer, AI Engineer, LLM Engineer).
Mark is_relevant=False for: pure BI/reporting with no engineering, data entry, database admin
with no ML component.
""",
    "any": """
You are a generalist recruiter. Accept all professional roles. Set is_relevant=True for any
non-spam, legitimate job posting. Mark is_relevant=False only for obvious spam, duplicate postings,
or completely irrelevant roles (e.g. manual labour when searching for tech roles).
""",
}

EXTRACTION_PROMPT_TEMPLATE = """
{domain_instructions}

### EXTRACTION (for ALL roles, regardless of relevancy):
- `is_relevant`: Boolean (True if relevant per domain definition above).
- `tech_stack`: List of SPECIFIC tools, platforms, or technologies mentioned.
- `seniority`: Intern, Junior, Mid, Senior, Staff, Lead, Head, Director, VP, or Manager.
- `employment_type`: Full-time, Part-time, Contract, Freelance, or Internship.
- `min_years_exp`: Minimum years of experience required (Integer, 0 if not specified).
- `salary_min`: Minimum base salary (USD equivalent, numeric, null if not mentioned).
- `salary_max`: Maximum base salary (USD equivalent, numeric, null if not mentioned).
- `reason`: One sentence explaining the relevancy decision.

### INPUT DATA:
{jobs_chunk}

### OUTPUT FORMAT:
Return a JSON array of objects, one per job in the same order. Use the following schema:
[
  {{
    "id": "original_id",
    "is_relevant": true,
    "tech_stack": ["tool1", "tool2"],
    "seniority": "Senior",
    "employment_type": "Full-time",
    "min_years_exp": 5,
    "salary_min": 150000,
    "salary_max": 200000,
    "reason": "..."
  }}
]
"""

def extract_structured_data(raw_data: list[dict], domain: str = "any", model: str = "gpt-4o-mini", batch_size: int = 20) -> list[dict]:
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    if not client.api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set.")
        
    domain_instructions = DOMAIN_PROMPTS.get(domain, DOMAIN_PROMPTS["any"])
    
    all_results = []
    
    for i in range(0, len(raw_data), batch_size):
        batch = raw_data[i:i+batch_size]
        jobs_chunk = []
        for j, job in enumerate(batch):
            jobs_chunk.append({
                "idx": j,
                "title": job.get("title", ""),
                "company": job.get("company", ""),
                "description": (job.get("description", "") or "")[:4000]
            })
            
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a precise data extraction specialist."},
                    {"role": "user", "content": EXTRACTION_PROMPT_TEMPLATE.format(
                        domain_instructions=domain_instructions,
                        jobs_chunk=json.dumps(jobs_chunk, indent=2)
                    )}
                ],
                response_format={"type": "json_object"}
            )
            data = json.loads(response.choices[0].message.content)
            
            extracted = []
            if isinstance(data, dict):
                for val in data.values():
                    if isinstance(val, list):
                        extracted = val
                        break
                if not extracted:
                    extracted = [data]
            else:
                extracted = data
                
            for j, res in enumerate(extracted):
                item = batch[j].copy()
                if res:
                    # stringify list to store in CSV natively
                    if "tech_stack" in res and isinstance(res["tech_stack"], list):
                        res["tech_stack"] = json.dumps(res["tech_stack"])
                    item.update(res)
                all_results.append(item)
                
        except Exception as e:
            logging.error(f"Error in extraction batch: {e}")
            all_results.extend(batch)
            
        time.sleep(0.5)
        
    return all_results


async def extract_structured_data_async(raw_data: list[dict], domain: str = "any", model: str = "gpt-4o-mini", batch_size: int = 20) -> list[dict]:
    """Async wrapper for the blocking OpenAI extraction logic."""
    import asyncio
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        extract_structured_data,
        raw_data, domain, model, batch_size
    )


DOMAIN_KEYWORD_MAP = {
    "gtm": ["gtm", "revops", "revenue operations", "growth engineer", "sales engineer"],
    "sales": ["account executive", "sales director", "channel", "partner manager",
              "business development", "solutions consultant", "sales manager"],
    "biotech": ["bioinformatics", "computational biology", "genomics", "pharma",
                "life sciences", "clinical", "immunology", "drug discovery"],
    "data": ["data scientist", "ml engineer", "machine learning", "data engineer",
             "analytics engineer", "ai engineer", "llm engineer"],
}

def detect_domain(keywords: list[str]) -> str:
    """Infer job_domain from keywords. Returns 'any' if ambiguous."""
    kw_lower = " ".join(keywords).lower()
    for domain, signals in DOMAIN_KEYWORD_MAP.items():
        if any(s in kw_lower for s in signals):
            return domain
    return "any"


def build_urls(keywords: list[str], locations: list[str]) -> dict[str, list[str]]:
    """Returns {source_name: [url1, url2, ...]} grouped by scraping source."""
    urls_by_source = {"builtin": [], "linkedin": []}

    for kw in keywords:
        for loc in locations:
            kw_url = kw.replace(" ", "+")
            loc_url = loc.replace(" ", "+")

            # Determine source by location
            if any(loc.lower() in l.lower() for l in SOURCE_MAP["linkedin"]):
                url = f"https://www.linkedin.com/jobs/search/?keywords={kw_url}&location={loc_url}"
                urls_by_source["linkedin"].append(url)
            else:
                url = f"https://builtin.com/jobs?search={kw_url}&location={loc_url}"
                urls_by_source["builtin"].append(url)

    return urls_by_source


async def scrape(
    keywords: list[str],
    locations: list[str],
    max_per_query: int = 100,
    job_domain: str | None = None,
) -> pd.DataFrame:
    """
    Returns a DataFrame with the canonical schema columns defined in TASK.md.
    Deduplication on (title, company, url) should be applied before returning.
    """
    job_domain = job_domain or detect_domain(keywords)
    logging.info(f"Using job_domain: {job_domain}")
    
    urls_by_source = build_urls(keywords, locations)
    
    # Stage 1: Async Apify Extraction
    logging.info(f"Starting parallel Apify scrape...")
    raw_results = await scrape_apify(urls_by_source, max_per_query)
    logging.info(f"Apify collected {len(raw_results)} unique items. Proceeding to LLM extraction...")
    
    # Stage 2: Async OpenAI Structure Extraction
    if raw_results:
        structured_results = await extract_structured_data_async(raw_results, domain=job_domain)
    else:
        structured_results = []
        
    df = pd.DataFrame(structured_results)
    
    # Ensure all required canonical columns exist
    canonical_columns = [
        "title", "company", "category", "location", "date_posted", "description_html",
        "description_text", "salary_json_min", "salary_json_max", "salary_json_currency",
        "salary_json_unit", "hiring_remote_in", "workplace_type", "salary_range_short",
        "seniority", "employment_type", "workplace_type_enum", "company_overview", "url", "source",
        "description", "postedDate", "salary", "id", "is_relevant", "tech_stack",
        "min_years_exp", "salary_min", "salary_max", "reason"
    ]
    
    for col in canonical_columns:
        if col not in df.columns:
            df[col] = ""
            
    if not df.empty:
        df = df.drop_duplicates(subset=["title", "company", "url"]).reset_index(drop=True)
        
    return df
