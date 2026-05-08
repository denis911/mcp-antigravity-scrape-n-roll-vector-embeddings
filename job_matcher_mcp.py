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
from scrapers.builtin_scraper import scrape_builtin, DEFAULT_SOURCE_MAP
from scrapers.serper_scraper import scrape_serper
from scrapers.japan_scraper import scrape_japan, DEFAULT_JAPAN_LOCATIONS
from scrapers.linkedin_scraper import scrape_linkedin

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

async def _run_scrape(
    keywords: list[str],
    locations: list[str],
    max_results_per_query: int = 100,
    job_domain: str | None = None,
    max_total_queries: int | None = None,
    output_path: str | None = None,
    source_map: dict | None = None,
    ats_domains: list[str] | None = None,
) -> pd.DataFrame:
    """Internal shared scraping logic."""
    max_total_queries = max_total_queries or int(os.getenv("MAX_TOTAL_QUERIES", 8))
    
    # Safety cap
    total_requested = len(keywords) * len(locations)
    if total_requested > max_total_queries:
        logging.warning(f"Total queries {total_requested} exceeds safety cap {max_total_queries}. Truncating.")
        flat_queries = [(kw, loc) for kw in keywords for loc in locations][:max_total_queries]
        keywords = sorted(list(set(q[0] for q in flat_queries)))
        locations = sorted(list(set(q[1] for q in flat_queries)))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = output_path or f"data/raw_jobs_{timestamp}.csv"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    logging.info(f"Scraping jobs for keywords={keywords}, locations={locations}, domain={job_domain}")
    
    s_map = source_map or DEFAULT_SOURCE_MAP
    builtin_locs = s_map.get("builtin", [])
    linkedin_locs = s_map.get("linkedin", [])
    
    us_locations = [l for l in locations if any(l.lower() in x.lower() for x in builtin_locs)]
    jp_locations = [l for l in locations if any(l.lower() in j.lower() for j in DEFAULT_JAPAN_LOCATIONS)]
    li_locations = [l for l in locations if any(l.lower() in x.lower() for x in linkedin_locs) and l not in jp_locations]
    eu_locations = [l for l in locations if l not in us_locations and l not in jp_locations and l not in li_locations]

    dfs = []
    if us_locations:
        df_us = await scrape_builtin(keywords, us_locations, max_results_per_query, job_domain=job_domain, source_map=source_map)
        if not df_us.empty:
            dfs.append(df_us)
    if li_locations:
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
            
    if dfs:
        df = pd.concat(dfs, ignore_index=True)
    else:
        df = pd.DataFrame()
        
    if not df.empty:
        df = normalise_columns(df)
        df.to_csv(out_path, index=False)
        logging.info(f"Scraped {len(df)} jobs → {out_path}")
        
    return df

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
    Automatically routes locations to their most optimal scraper in a single run:
    - US locations (New York, SF, Boston, etc.) -> BuiltIn (Apify)
    - EU/Global locations (London, Berlin, etc.) -> LinkedIn (Apify)
    - Japanese locations (Japan, Tokyo, etc.) -> Japanese job boards (Serper)
    - All other locations -> Google Serper (ATS footprints)

    Results from all active scrapers are seamlessly merged, normalized, and saved to a single CSV.

    Parameters:
    - keywords: List of job titles or keywords (e.g., ["Data Scientist", "ML Engineer"]).
    - locations: List of locations (e.g., ["Berlin", "London", "New York", "Japan"]).
    - source_map: (Optional) Override routing for specific locations. 
      Format: {"builtin": ["Austin"], "linkedin": ["Dublin"], "serper": ["Paris"]}
    - ats_domains: (Optional) List of domain footprints for Serper search. 
      Example: ["boards.greenhouse.io", "wellfound.com", "thehub.io"]
    - job_domain: (Optional) Extraction focus: "gtm", "sales", "biotech", "data", "any".
      Auto-detected from keywords if not provided.
    - max_results_per_query: Max items per keyword×location pair (default: 100).
    - max_total_queries: Safety cap for total parallel requests (default from .env).

    Example Usage:
    scrape_jobs(
        keywords=["AI Engineer"],
        locations=["San Francisco", "London", "Japan"],
        ats_domains=["greenhouse.io", "lever.co"]
    )
    """
    DEFAULT_KEYWORDS = ["GTM engineer"]
    DEFAULT_LOCATIONS = ["Berlin", "London", "New York", "San Francisco", "Boston", "US remote"]
    
    keywords = keywords or DEFAULT_KEYWORDS
    locations = locations or DEFAULT_LOCATIONS

    df = await _run_scrape(
        keywords=keywords,
        locations=locations,
        max_results_per_query=max_results_per_query,
        job_domain=job_domain,
        max_total_queries=max_total_queries,
        output_path=output_path,
        source_map=source_map,
        ats_domains=ats_domains
    )
    
    return {
        "status": "ok",
        "csv_path": output_path or "data/raw_jobs_*.csv", # approximation if not provided
        "jobs_scraped": len(df),
        "queries_run": len(keywords) * len(locations) # approximation
    }

@mcp.tool()
async def compare_searches(
    search_strategies: list[dict],
    profile_path: str | None = None,
    max_results_per_query: int = 10,
    top_n_per_strategy: int = 5,
    output_dir: str | None = None,
) -> dict:
    """
    Run multiple keyword strategies, score each independently, return comparative stats.

    Example input:
    [
        {
            "name": "traditional_sales",
            "keywords": ["account executive", "channel manager"],
            "locations": ["London", "Berlin"],
            "job_domain": "sales"
        },
        {
            "name": "presales_se",
            "keywords": ["pre-sales engineer", "solutions engineer"],
            "locations": ["London", "Berlin"],
            "job_domain": "sales"
        }
    ]
    """
    profile_path = profile_path or os.getenv("DEFAULT_PROFILE_JSON")
    output_dir = output_dir or os.getenv("DEFAULT_OUTPUT_DIR", "output/")
    
    if not profile_path or not Path(profile_path).exists():
        raise FileNotFoundError(f"Profile not found at {profile_path}")
        
    with open(profile_path, encoding="utf-8") as f:
        profile = json.load(f)
    
    profile_text = profile_to_text(profile)
    model = get_model()
    profile_vec = model.encode([profile_text], normalize_embeddings=True)[0]

    results = []
    all_jobs = []

    for strategy in search_strategies:
        name = strategy["name"]
        keywords = strategy["keywords"]
        locations = strategy["locations"]
        job_domain = strategy.get("job_domain", "any")

        logging.info(f"compare_searches: running strategy '{name}'...")

        # Scrape
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = f"data/compare_{name}_{timestamp}.csv"
        df = await _run_scrape(
            keywords=keywords,
            locations=locations,
            max_results_per_query=max_results_per_query,
            job_domain=job_domain,
            output_path=csv_path
        )

        if df.empty:
            results.append({"name": name, "jobs_scraped": 0, "error": "no results"})
            continue

        # Embed & Score
        df["_clean_text"] = df.apply(get_job_text, axis=1)
        vecs = embed_texts(df["_clean_text"].tolist(), model)
        df["embedding"] = [json.dumps(v.tolist()) for v in vecs]
        df["embedding_model"] = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")

        scores = cosine_similarity([profile_vec], vecs)[0]
        df["similarity_score"] = scores
        df = df.sort_values("similarity_score", ascending=False).reset_index(drop=True)
        df = df.drop(columns=["_clean_text"], errors="ignore")
        df.to_csv(csv_path, index=False)

        # Stats
        valid_scores = scores[~np.isnan(scores)]
        dist = {
            ">=0.55": int((valid_scores >= 0.55).sum()),
            "0.45-0.55": int(((valid_scores >= 0.45) & (valid_scores < 0.55)).sum()),
            "<0.45": int((valid_scores < 0.45).sum()),
        }

        top_jobs = []
        for _, row in df.head(top_n_per_strategy).iterrows():
            job_info = {
                "title": str(row.get("title", "")),
                "company": str(row.get("company", "")),
                "score": round(float(row["similarity_score"]), 4),
                "url": str(row.get("url", "")),
                "strategy": name,
            }
            top_jobs.append(job_info)
            all_jobs.append(job_info)

        results.append({
            "name": name,
            "jobs_scraped": len(df),
            "median_score": round(float(np.median(valid_scores)), 4) if len(valid_scores) > 0 else 0,
            "mean_score": round(float(np.mean(valid_scores)), 4) if len(valid_scores) > 0 else 0,
            "max_score": round(float(valid_scores.max()), 4) if len(valid_scores) > 0 else 0,
            "score_distribution": dist,
            "top_jobs": top_jobs,
            "csv_path": csv_path,
        })

    # Find winner
    scored = [r for r in results if "median_score" in r]
    winner = max(scored, key=lambda x: x["median_score"])["name"] if scored else None

    # Merged top 10 across all strategies deduped by URL
    seen_urls = set()
    merged = []
    for job in sorted(all_jobs, key=lambda x: -x["score"]):
        if job["url"] not in seen_urls:
            seen_urls.add(job["url"])
            merged.append(job)
    merged_top10 = merged[:10]

    # Recommendation text
    if len(scored) >= 2:
        sorted_by_median = sorted(scored, key=lambda x: -x["median_score"])
        best = sorted_by_median[0]
        second = sorted_by_median[1]
        if second["median_score"] > 0:
            pct = round((best["median_score"] - second["median_score"]) / second["median_score"] * 100)
            recommendation = (
                f"'{best['name']}' returns {pct}% higher median score ({best['median_score']:.3f}) "
                f"than '{second['name']}' ({second['median_score']:.3f}). "
                f"Prioritise '{best['name']}' for future scrapes."
            )
        else:
            recommendation = f"'{best['name']}' is the only strategy with positive scores."
    else:
        recommendation = "Only one strategy returned results — no comparison available."

    return {
        "status": "ok",
        "strategies": results,
        "winner": winner,
        "recommendation": recommendation,
        "merged_top_10": merged_top10,
    }


@mcp.tool()
async def score_jobs(
    csv_path: str,
    embed_model: str | None = None,
    force_reembed: bool = False,
) -> dict:
    """
    Compute and cache job embeddings for a given CSV file.
    Uses local HuggingFace sentence-transformers (free, no API key).
    
    Incremental updates: Only embeds rows that don't have an embedding or 
    were embedded with a different model. Use `force_reembed=True` to refresh all.
    
    Parameters:
    - csv_path: Path to the CSV file (from scrape_jobs or list_saved_csvs).
    - embed_model: (Optional) Model name (default: "all-MiniLM-L6-v2").
    - force_reembed: If True, ignores cache and re-computes all embeddings.
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
    Score, rank, and explain top job matches against the candidate profile.
    
    Applies pre-filters (salary, seniority, relevancy) before computing cosine 
    similarity using local embeddings. Generates AI-powered fit explanations.

    Parameters:
    - csv_path: Path to the CSV file containing job postings.
    - profile_path: (Optional) Path to candidate profile JSON (e.g., "test_profile.json").
    - top_n: (Optional) Number of top matches to return (default: 10).
    - min_score: Minimum similarity score threshold (0.0 to 1.0).
    - explain: If True, generates 2-3 sentence fit explanations via LLM.
    - relevant_only: If True, filters out jobs marked as irrelevant by the extractor.
    - min_salary: Filter out jobs with base salary below this value.
    - exclude_seniority: List of seniority levels to exclude (e.g., ["Intern", "Junior"]).
    - employment_types: List of allowed employment types (e.g., ["Full-time"]).
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
async def snipe_url(
    url: str,
    profile_path: str | None = None,
    explain: bool = True,
) -> dict:
    """Fetch, parse, embed, and score a single job posting URL."""
    profile_path = profile_path or os.getenv("DEFAULT_PROFILE_JSON")
    if not profile_path or not Path(profile_path).exists():
        raise FileNotFoundError(f"Profile not found at {profile_path}")

    with open(profile_path, encoding="utf-8") as f:
        profile = json.load(f)

    # ── Step 1: Fetch URL content ─────────────────────────────────────────
    # Try direct HTTP first (fast, free). Apify fallback for blocked sites.
    description_text = ""
    title = ""

    try:
        import aiohttp
        from bs4 import BeautifulSoup
        async with aiohttp.ClientSession() as session:
            async with session.get(
                url,
                timeout=aiohttp.ClientTimeout(total=15),
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
            ) as resp:
                if resp.status == 200:
                    html = await resp.text()
                    soup = BeautifulSoup(html, "html.parser")
                    title_tag = soup.find("h1") or soup.find("title")
                    title = title_tag.get_text(strip=True) if title_tag else ""
                    for tag in soup(["script", "style", "nav", "header", "footer", "noscript"]):
                        tag.extract()
                    description_text = soup.get_text(separator=" ", strip=True)
                    logging.info(f"snipe_url: fetched {len(description_text)} chars via HTTP")
    except Exception as e:
        logging.warning(f"snipe_url: direct HTTP failed ({e}), trying Apify fallback")

    # Apify fallback for LinkedIn and other bot-protected sites
    if len(description_text) < 200:
        try:
            token = os.environ.get("APIFY_API_TOKEN")
            if token:
                loop = asyncio.get_event_loop()
                def _apify_fetch():
                    from apify_client import ApifyClient
                    client = ApifyClient(token)
                    # cheerio-scraper: cheap, fast, handles most job boards
                    run = client.actor("apify/cheerio-scraper").call(run_input={
                        "startUrls": [{"url": url}],
                        "maxCrawlingDepth": 0,
                        "maxPagesPerCrawl": 1,
                    })
                    items = list(client.dataset(run["defaultDatasetId"]).iterate_items())
                    if items:
                        item = items[0]
                        return (
                            item.get("text", "")
                            or item.get("pageContent", "")
                            or item.get("body", "")
                        )
                    return ""
                description_text = await loop.run_in_executor(None, _apify_fetch)
                logging.info(f"snipe_url: Apify fetched {len(description_text)} chars")
        except Exception as e:
            logging.error(f"snipe_url: Apify fallback failed: {e}")

    if len(description_text) < 100:
        return {
            "status": "error",
            "error": (
                f"Could not fetch sufficient content from {url}. "
                "Try copy-pasting the JD text directly."
            )
        }

    # ── Step 2: LLM extraction ────────────────────────────────────────────
    raw_item = {
        "title": title,
        "company": "",
        "url": url,
        "description_text": description_text[:6000],
        "description": description_text[:6000],
        "source": "snipe",
    }

    from extractor import extract_structured_data_async, ensure_canonical_columns
    structured = await extract_structured_data_async([raw_item], domain="any")
    if not structured:
        return {"status": "error", "error": "LLM extraction returned no results"}

    item = structured[0]
    df_row = ensure_canonical_columns(pd.DataFrame([item])).iloc[0]

    # ── Step 3: Embed ─────────────────────────────────────────────────────
    text = get_job_text(df_row)
    model = get_model()
    vec = model.encode([text], normalize_embeddings=True)[0]

    # ── Step 4: Score against profile ────────────────────────────────────
    profile_text = profile_to_text(profile)
    profile_vec = model.encode([profile_text], normalize_embeddings=True)[0]
    score = float(cosine_similarity([profile_vec], [vec])[0][0])

    # ── Step 5: Fit explanation ───────────────────────────────────────────
    fit_explanation = ""
    if explain:
        try:
            client_oai = AsyncOpenAI()
            llm_model = os.getenv("OPENAI_EXPLAIN_MODEL", "gpt-4o-mini")
            candidate_summary = profile.get("summary", {}).get("elevator_pitch", "")
            fit_explanation = await _explain_job(
                client_oai, llm_model, candidate_summary, df_row
            )
        except Exception as e:
            logging.warning(f"snipe_url: explanation failed: {e}")

    return {
        "status": "ok",
        "title": str(item.get("title", title)),
        "company": str(item.get("company", "")),
        "location": str(item.get("location", "")),
        "salary": str(item.get("salary", "")),
        "seniority": str(item.get("seniority", "")),
        "employment_type": str(item.get("employment_type", "")),
        "tech_stack": item.get("tech_stack", []),
        "min_years_exp": item.get("min_years_exp", ""),
        "is_relevant": item.get("is_relevant", ""),
        "url": url,
        "similarity_score": round(score, 4),
        "fit_explanation": fit_explanation,
        "description_text": description_text[:1500],
        "source": "snipe",
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
