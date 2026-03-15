import os
import json
import pandas as pd
from openai import OpenAI
from tqdm import tqdm
import time

# --- CONFIGURATION ---
INPUT_FILE = "_internal/apify_raw_export.json"
OUTPUT_JSON = "_internal/structured_jobs.json"
OUTPUT_CSV = "_internal/structured_jobs.csv"
MODEL = "gpt-4o-mini"
BATCH_SIZE = 20 # Increased for production run

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

GTM_TOOLS = [
    "Salesforce", "HubSpot", "Clay", "n8n", "Tray.io", "Zapier", "Segment", 
    "mParticle", "Hightouch", "Census", "Marketo", "Pardot", "Outreach", 
    "Salesloft", "Amplemarket", "Intercom", "Drift", "FullStory", "Amplitude", 
    "Mixpanel", "Snowflake", "BigQuery", "Looker", "Tableau", "dbt", "Python", 
    "SQL", "Javascript", "Typescript", "React", "Node.js"
]

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
  }},
  ...
]
"""

def extract_jobs(batch):
    # Prepare chunk for prompt
    jobs_chunk = []
    for i, job in enumerate(batch):
        jobs_chunk.append({
            "idx": i,
            "title": job.get("title", ""),
            "company": job.get("company", ""),
            "description": (job.get("description", "") or "")[:4000] # Truncate to save tokens
        })
    
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a precise data extraction specialist."},
                {"role": "user", "content": EXTRACTION_PROMPT.format(jobs_chunk=json.dumps(jobs_chunk, indent=2))}
            ],
            response_format={"type": "json_object"}
        )
        # Handle the fact that we return a list in the prompt but response_format expects an object wrapper usually
        res_text = response.choices[0].message.content
        data = json.loads(res_text)
        
        # In case the LLM wraps it in a key
        if isinstance(data, dict):
            # Try to find a list value
            for val in data.values():
                if isinstance(val, list):
                    return val
            return [data] # fallback
        return data
    except Exception as e:
        print(f"Error in batch: {e}")
        return [None] * len(batch)

def main(test_mode=False):
    if not os.path.exists(INPUT_FILE):
        print(f"Input file {INPUT_FILE} not found.")
        return

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    if test_mode:
        print("⚙️ TEST MODE: Processing first 10 items only...")
        raw_data = raw_data[:10]

    all_results = []
    
    print(f"🚀 Processing {len(raw_data)} jobs in batches of {BATCH_SIZE}...")
    
    for i in tqdm(range(0, len(raw_data), BATCH_SIZE)):
        batch = raw_data[i:i+BATCH_SIZE]
        results = extract_jobs(batch)
        
        # Merge structured data back with basic metadata
        for j, res in enumerate(results):
            if res and i + j < len(raw_data):
                full_item = raw_data[i+j].copy()
                full_item.update(res)
                all_results.append(full_item)
            elif i + j < len(raw_data):
                # Fallback if LLM failed
                all_results.append(raw_data[i+j])
        
        # Simple rate limiting for safety
        time.sleep(0.5)

    # Convert to DataFrame for final stats and saving
    df = pd.DataFrame(all_results)
    
    # Save results
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2)
    
    df.to_csv(OUTPUT_CSV, index=False)
    
    # Print Summary
    print("\n--- STAGE 2 SUMMARY ---")
    if "is_gtm_technical" in df.columns:
        valid_count = df[df["is_gtm_technical"] == True].shape[0]
        print(f"✅ GTM Technical Roles: {valid_count} / {len(df)}")
    
    print(f"📄 Saved results to {OUTPUT_JSON} and {OUTPUT_CSV}")

if __name__ == "__main__":
    import sys
    is_test = "--test" in sys.argv
    main(test_mode=is_test)
