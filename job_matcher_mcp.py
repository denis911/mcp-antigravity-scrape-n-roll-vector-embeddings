import os
import sys
import json
import asyncio
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from openai import AsyncOpenAI

from job_scorer import get_job_text, profile_to_text, embed_texts
from builtin_scraper import scrape_builtin, DEFAULT_SOURCE_MAP
from serper_scraper import scrape_serper
from japan_scraper import scrape_japan, DEFAULT_JAPAN_LOCATIONS

# ── Configuration & Initialization ──────────────────────────────────────────

load_dotenv()  # loads .env; does NOT override Windows env vars automatically

# Logging to stderr only — stdout is reserved for MCP protocol
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

mcp = FastMCP("job-matcher")

# Global singleton for the embedding model (loaded at runtime)
_model: SentenceTransformer | None = None

def get_model() -> SentenceTransformer:
    """Lazy load the embedding model to avoid slowing down imports/tests."""
    global _model
    if _model is None:
        model_name = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
        logging.info(f"Loading embedding model '{model_name}'...")
        _model = SentenceTransformer(model_name)
        logging.info("Embedding model ready.")
    return _model

# ── Data Normalisation ───────────────────────────────────────────────────────

COLUMN_RENAME_MAP = {
    "salary_json_min": "salary_min",
    "salary_json_max": "salary_max",
    "salary_json_currency": "salary_currency",
    "salary_json_unit": "salary_unit",
    "workplace_type_enum": "workplace_type",
    "is_gtm_technical": "is_relevant",
}

def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise DataFrame columns per TASK.md schema.
    Applies immediately after pd.read_csv().
    """
    # 1. Strip whitespace from column names
    df.columns = df.columns.str.strip()
    # 2. Lowercase + replace spaces with underscores
    df.columns = df.columns.str.lower().str.replace(" ", "_", regex=False)
    # 3. Apply explicit rename map
    df = df.rename(columns=COLUMN_RENAME_MAP)
    # 4. Remove duplicate columns (keep first occurrence)
    df = df.loc[:, ~df.columns.duplicated(keep="first")]
    return df

# ── MCP Tools ───────────────────────────────────────────────────────────────

@mcp.tool()
async def scrape_jobs(
    keywords: list[str] | None = None,
    locations: list[str] | None = None,
    max_results_per_query: int = 100,
    job_domain: str | None = None,
    max_total_queries: int | None = None,
    output_path: str | None = None,
    source_map: dict | None = None,
    ats_domains: list[str] | None = None,
) -> dict:
    """
    Scrape job postings matching keywords × locations.
    Automatically routes US locations to BuiltIn (via Apify) and EU/Other locations to Google Serper (ATS search) in a single run.
    Results from both scrapers are merged into a single CSV.

    Parameters:
    - keywords: List of job titles or keywords (e.g., ["Data Scientist", "ML Engineer"]).
    - locations: List of locations to search in (e.g., ["Berlin", "London", "New York"]).
    - source_map: (Optional) A dictionary overriding the routing of locations to scrapers. 
      Format: {"builtin": ["New York", "Boston"], "linkedin": ["Berlin", "London"]}
      Locations mapped to 'builtin' will use the Apify builtin scraper. All other locations will default to Serper.
    - ats_domains: (Optional) A list of domain footprints to restrict Google ATS search (used for Serper routing). 
      Useful for EU startups. Example: ["wellfound.com", "ycombinator.com/jobs", "thehub.io", "topstartups.io"]
    - locations: Special values that trigger Japanese board routing (via japan_scraper.py):
      "Japan", "Tokyo", "Japanese companies", "Japan remote".
      These locations route to daijob.com, gaijinpot.com, careercross.com, jp.japanese-jobs.com instead of BuiltIn or Serper. Location string is NOT appended to queries for these boards.

    Example Usage:
    scrape_jobs(
        keywords=["AI Engineer"],
        locations=["San Francisco", "London", "Japan"],
        ats_domains=["greenhouse.io", "lever.co"]
    )
    # "San Francisco" is automatically routed to BuiltIn (Apify).
    # "London" is automatically routed to Serper (Google ATS).
    # "Japan" is automatically routed to Japan Scraper (Japanese Job Boards).
    """
    DEFAULT_KEYWORDS = ["GTM engineer"]
    DEFAULT_LOCATIONS = ["Berlin", "London", "New York", "San Francisco", "Boston", "US remote"]
    
    max_total_queries = max_total_queries or int(os.getenv("MAX_TOTAL_QUERIES", 8))
    
    keywords = keywords or DEFAULT_KEYWORDS
    locations = locations or DEFAULT_LOCATIONS

    # Safety cap
    total_requested = len(keywords) * len(locations)
    if total_requested > max_total_queries:
        logging.warning(f"Total queries {total_requested} exceeds safety cap {max_total_queries}. Truncating.")
        # Simple truncation: take first N combinations
        flat_queries = [(kw, loc) for kw in keywords for loc in locations][:max_total_queries]
        keywords = sorted(list(set(q[0] for q in flat_queries)))
        locations = sorted(list(set(q[1] for q in flat_queries)))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = output_path or f"data/raw_jobs_{timestamp}.csv"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    logging.info(f"Scraping jobs for keywords={keywords}, locations={locations}, domain={job_domain}")
    
    s_map = source_map or DEFAULT_SOURCE_MAP
    builtin_locs = s_map.get("builtin", [])
    
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
            
    if dfs:
        df = pd.concat(dfs, ignore_index=True)
    else:
        df = pd.DataFrame()
        
    if not df.empty:
        df = normalise_columns(df)
        
    df.to_csv(out_path, index=False)
    logging.info(f"Scraped {len(df)} jobs → {out_path}")
    
    return {
        "status": "ok",
        "csv_path": out_path,
        "jobs_scraped": len(df),
        "queries_run": min(total_requested, max_total_queries)
    }


@mcp.tool()
async def score_jobs(
    csv_path: str,
    embed_model: str | None = None,
    force_reembed: bool = False,
) -> dict:
    """
    Compute and cache job embeddings.
    """
    df = pd.read_csv(csv_path)
    df = normalise_columns(df)
    
    model = get_model()
    model_name = embed_model or os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
    
    if "embedding" not in df.columns:
        df["embedding"] = None
    if "embedding_model" not in df.columns:
        df["embedding_model"] = None
        
    mask = df["embedding"].isna() | (df["embedding_model"] != model_name)
    if force_reembed:
        mask = pd.Series(True, index=df.index)
        
    to_embed = df[mask]
    if not to_embed.empty:
        logging.info(f"Embedding {len(to_embed)} new records...")
        df.loc[mask, "_clean_text"] = to_embed.apply(get_job_text, axis=1)
        embeddings = embed_texts(df.loc[mask, "_clean_text"].tolist(), model)
        df.loc[mask, "embedding"] = [json.dumps(e.tolist()) for e in embeddings]
        df.loc[mask, "embedding_model"] = model_name
        df = df.drop(columns=["_clean_text"], errors="ignore")
    
    df.to_csv(csv_path, index=False)
    
    return {
        "status": "ok",
        "csv_path": csv_path,
        "embedded": len(to_embed),
        "skipped_cached": len(df) - len(to_embed),
        "model_used": model_name
    }


async def _explain_job(client: AsyncOpenAI, model: str, candidate_summary: str, row: pd.Series) -> str:
    EXPLAIN_PROMPT = """\
You are a career advisor. Given a candidate profile summary and a job posting,
write 2-3 sentences explaining why this job IS or IS NOT a good fit.
Be specific: mention exact skill overlaps, domain matches, or gaps.
No bullet points. Plain text only.

Candidate: {candidate_summary}

Job: {title} at {company} ({location})
Similarity score: {score:.3f}
Description: {description_snippet}
Tech stack: {tech_stack}
"""
    try:
        snippet = str(row.get("description_text", ""))[:600]
        prompt = EXPLAIN_PROMPT.format(
            candidate_summary=candidate_summary,
            title=row.get("title", ""),
            company=row.get("company", ""),
            location=row.get("location", ""),
            score=row.get("similarity_score", 0.0),
            description_snippet=snippet,
            tech_stack=row.get("tech_stack", "")
        )
        
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Error explaining fit for {row.get('title')}: {e}")
        return f"Error generating explanation: {e}"


@mcp.tool()
async def get_top_jobs(
    csv_path: str,
    profile_path: str | None = None,
    top_n: int | None = None,
    min_score: float = 0.0,
    explain: bool = True,
    output_dir: str | None = None,
    # NEW FILTERS with defaults from env:
    exclude_seniority: list[str] | None = None,
    employment_types: list[str] | None = None,
    min_salary: float | None = None,
    relevant_only: bool = True,
) -> dict:
    """
    Score, rank, explain, save, return top matches.
    """
    df = pd.read_csv(csv_path)
    df = normalise_columns(df)
    original_count = len(df)
    
    # ── Resolve Defaults ───────────────────────────────────────────────────
    if exclude_seniority is None:
        raw = os.getenv("DEFAULT_EXCLUDE_SENIORITY", "")
        exclude_seniority = [s.strip() for s in raw.split(",") if s.strip()]
        
    if employment_types is None:
        raw = os.getenv("DEFAULT_EMPLOYMENT_TYPE", "")
        employment_types = [e.strip() for e in raw.split(",") if e.strip()]
        
    if min_salary is None:
        min_salary = float(os.getenv("DEFAULT_MIN_SALARY", 0))

    # ── Pre-filters ────────────────────────────────────────────────────────
    if relevant_only and "is_relevant" in df.columns:
        # Explicitly require "true" — treats NaN and empty as not relevant
        df = df[df["is_relevant"].astype(str).str.lower() == "true"]
    
    if exclude_seniority:
        excl = [s.lower() for s in exclude_seniority]
        df = df[~df["seniority"].astype(str).str.lower().isin(excl)]
        
    if employment_types:
        emp = [e.lower() for e in employment_types]
        df = df[df["employment_type"].astype(str).str.lower().isin(emp)]
        
    if min_salary and "salary_min" in df.columns:
        df = df[pd.to_numeric(df["salary_min"], errors="coerce").fillna(0) >= min_salary]
        
    logging.info(f"After pre-filters: {len(df)} rows remain from {original_count}")
    
    if df.empty:
        return {
            "status": "ok",
            "message": "No jobs match the given filters.",
            "total_scored": 0,
            "returned": 0,
            "top_jobs": []
        }

    if "embedding" not in df.columns or df["embedding"].isna().all():
        raise ValueError("No embeddings found. Run score_jobs() first.")
        
    profile_path = profile_path or os.getenv("DEFAULT_PROFILE_JSON")
    if not profile_path or not Path(profile_path).exists():
        raise FileNotFoundError(f"Profile not found at {profile_path}")
        
    with open(profile_path, encoding="utf-8") as f:
        profile = json.load(f)
        
    profile_text = profile_to_text(profile)
    model = get_model()
    profile_vec = model.encode([profile_text], normalize_embeddings=True)[0]
    
    valid = df["embedding"].notna()
    job_vecs = np.array([json.loads(e) for e in df.loc[valid, "embedding"]])
    
    scores = cosine_similarity([profile_vec], job_vecs)[0]
    df.loc[valid, "similarity_score"] = scores
    
    df = df.sort_values("similarity_score", ascending=False).reset_index(drop=True)
    if min_score > 0:
        df = df[df["similarity_score"] >= min_score]
        
    effective_top_n = top_n or int(os.getenv("DEFAULT_TOP_N", 10))
    df_top = df.head(effective_top_n).copy()
    
    if explain:
        if "fit_explanation" not in df_top.columns:
            df_top["fit_explanation"] = None
            
        mask = df_top["fit_explanation"].isna()
        if mask.any():
            client = AsyncOpenAI()
            llm_model = os.getenv("OPENAI_EXPLAIN_MODEL", "gpt-4o-mini")
            candidate_summary = profile.get("summary", {}).get("elevator_pitch", "")
            
            tasks = [
                _explain_job(client, llm_model, candidate_summary, row)
                for _, row in df_top[mask].iterrows()
            ]
            explanations = await asyncio.gather(*tasks)
            df_top.loc[mask, "fit_explanation"] = explanations
            
    # Save ranked CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = output_dir or os.getenv("DEFAULT_OUTPUT_DIR", "output/")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(out_dir) / f"jobs_ranked_{timestamp}.csv"
    
    df_out = df_top.drop(columns=["embedding", "embedding_model"], errors="ignore")
    df_out.to_csv(out_path, index=False)
    
    # Format return dictionary
    ret_cols = [
        "title", "company", "location", "salary", "seniority", "employment_type",
        "workplace_type", "tech_stack", "min_years_exp", "url", 
        "similarity_score"
    ]
    if explain:
        ret_cols.append("fit_explanation")
        
    # intersect with existing columns safely
    ret_cols = [c for c in ret_cols if c in df_top.columns]
    
    top_jobs_json = df_top[ret_cols].to_dict(orient="records")
    
    return {
        "status": "ok",
        "output_csv": str(out_path),
        "total_scored": len(valid) if sum(valid) else 0,
        "returned": len(df_top),
        "score_range": {
            "max": float(scores.max()) if len(scores) else 0.0,
            "min": float(scores.min()) if len(scores) else 0.0,
            "median": float(np.median(scores)) if len(scores) else 0.0,
        },
        "top_jobs": top_jobs_json
    }


@mcp.tool()
async def list_saved_csvs() -> dict:
    """
    List all CSV files in data/ and output/ directories.
    Useful for inspecting scraped or ranked data.
    """
    data_dir = os.getenv("DEFAULT_DATA_DIR", "data/")
    out_dir = os.getenv("DEFAULT_OUTPUT_DIR", "output/")
    
    files = []
    for d in [data_dir, out_dir]:
        path = Path(d)
        if path.exists():
            for f in path.glob("*.csv"):
                stats = f.stat()
                files.append({
                    "name": f.name,
                    "location": path.name,
                    "path": str(f),
                    "size_kb": round(stats.st_size / 1024, 2),
                    "modified": datetime.fromtimestamp(stats.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
                })
                
    return {
        "status": "ok",
        "files": sorted(files, key=lambda x: x["modified"], reverse=True)
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")
