import os
import json
import logging
import time
import pandas as pd
from apify_client import ApifyClient
from openai import OpenAI

ACTOR_PRIMARY = "IhQuCmT40q1tetuv3"  # easyapi/builtin-jobs-scraper ($19.99/mo)
ACTOR_FALLBACK = "shahidirfan/builtin-jobs-scraper"  # pay-per-usage only

def scrape_apify(start_urls: list[str], max_items: int) -> list[dict]:
    token = os.environ.get("APIFY_API_TOKEN")
    if not token:
        raise ValueError("APIFY_API_TOKEN environment variable is not set.")
        
    client = ApifyClient(token)
    all_results = []
    
    for i, url in enumerate(start_urls, 1):
        logging.info(f"Scraping URL {i}/{len(start_urls)}: {url}")
        run_input = {
            "startUrl": url,
            "results_wanted": max_items,
            "max_pages": 100
        }
        try:
            try:
                logging.info(f"Calling primary actor: {ACTOR_PRIMARY}")
                run = client.actor(ACTOR_PRIMARY).call(run_input=run_input)
            except Exception as e:
                error_msg = str(e).lower()
                if "subscription" in error_msg or "payment" in error_msg or "paid" in error_msg or "403" in error_msg:
                    logging.warning(f"Primary actor restricted. Falling back to {ACTOR_FALLBACK}")
                    run = client.actor(ACTOR_FALLBACK).call(run_input=run_input)
                else:
                    raise

            dataset_id = run["defaultDatasetId"]
            items = list(client.dataset(dataset_id).iterate_items())
            logging.info(f"Got {len(items)} items from {url}")
            all_results.extend(items)
        except Exception as e:
            logging.error(f"Error scraping {url}: {e}")
            
    # Dedup by URL
    seen_urls = set()
    results = []
    for item in all_results:
        item_url = item.get("url")
        if item_url and item_url not in seen_urls:
            seen_urls.add(item_url)
            results.append(item)
            
    # Normalize missing fields mapped by different actors
    for item in results:
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

        if not item.get("workType") and item.get("workplace_type"):
            item["workType"] = item["workplace_type"]

        if not item.get("experienceLevel") and item.get("seniority"):
            item["experienceLevel"] = item["seniority"]

        if not item.get("category_raw") and item.get("category"):
            item["category_raw"] = item["category"]

    return results

def extract_structured_data(raw_data: list[dict], model: str = "gpt-4o-mini", batch_size: int = 20) -> list[dict]:
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    if not client.api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set.")
        
    EXTRACTION_PROMPT = """
You are a GTM (Go-To-Market) Engineering expert. Analyze the following job descriptions and extract structured data.

### OBJECTIVE:
1. **Relevancy Filter**: Identify if the role is a "Technical GTM" role (e.g., GTM Engineer, Growth Engineer, Sales Engineer, RevOps Engineer, Fullstack Growth). 
   - Skip pure generic Software Engineering roles that have zero mention of revenue tools, CRM integrations, or growth experiments.
   - Skip pure non-technical Sales/Marketing roles (e.g., BDR, Account Executive, Content Manager).

2. **Extraction**:
   - `is_gtm_technical`: Boolean (True if relevant).
   - `tech_stack`: List of SPECIFIC GTM or data tools from the text (e.g. Clay, n8n, Salesforce, HubSpot, dbt, etc.). 
   - `seniority`: Junior, Mid, Senior, Staff, Lead, Head, or Manager.
   - `min_years_exp`: Minimum years of experience required (Integer).
   - `salary_min`: Minimum base salary (USD, numeric).
   - `salary_max`: Maximum base salary (USD, numeric).
   - `reason`: Brief explanation for the relevancy decision.

### INPUT DATA:
{jobs_chunk}

### OUTPUT FORMAT:
Return a JSON array of objects, one per job in the same order. Use the following schema:
[
  {{
    "id": "original_id",
    "is_gtm_technical": true,
    "tech_stack": ["tool1", "tool2"],
    "seniority": "Senior",
    "min_years_exp": 5,
    "salary_min": 150000,
    "salary_max": 200000,
    "reason": "..."
  }}
]
"""
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
                    {"role": "user", "content": EXTRACTION_PROMPT.format(jobs_chunk=json.dumps(jobs_chunk, indent=2))}
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


def scrape(
    keywords: list[str],
    locations: list[str],
    max_per_query: int = 100,
) -> pd.DataFrame:
    """
    Returns a DataFrame with the canonical schema columns defined in TASK.md.
    Deduplication on (title, company, url) should be applied before returning.
    """
    # Build BuiltIn start URLs from keywords & locations
    # (BuiltIn typically relies on query params, mapping simple keywords for demonstration.
    # In a real heavy use case, this maps exactly to BuiltIn search schema)
    # E.g. https://builtin.com/jobs?search=GTM+Engineer&location=Berlin
    start_urls = []
    for kw in keywords:
        for loc in locations:
            kw_url = kw.replace(' ', '+')
            loc_url = loc.replace(' ', '+')
            start_urls.append(f"https://builtin.com/jobs?search={kw_url}&location={loc_url}")
            
    # Stage 1: Apify Extraction
    logging.info(f"Starting Apify scrape for {len(start_urls)} queries...")
    raw_results = scrape_apify(start_urls, max_per_query)
    logging.info(f"Apify collected {len(raw_results)} unique items. Proceeding to LLM extraction...")
    
    # Stage 2: OpenAI Structure Extraction
    if raw_results:
        structured_results = extract_structured_data(raw_results)
    else:
        structured_results = []
        
    df = pd.DataFrame(structured_results)
    
    # Ensure all required canonical columns exist
    canonical_columns = [
        "title", "company", "category", "location", "date_posted", "description_html",
        "description_text", "salary_json_min", "salary_json_max", "salary_json_currency",
        "salary_json_unit", "hiring_remote_in", "workplace_type", "salary_range_short",
        "seniority", "workplace_type_enum", "company_overview", "url", "source",
        "description", "postedDate", "salary", "id", "is_gtm_technical", "tech_stack",
        "min_years_exp", "salary_min", "salary_max", "reason"
    ]
    
    for col in canonical_columns:
        if col not in df.columns:
            df[col] = ""
            
    if not df.empty:
        df = df.drop_duplicates(subset=["title", "company", "url"]).reset_index(drop=True)
        
    return df
