# TASK.md — Job Matcher MCP Server

## Goal

Build a local MCP server (`job_matcher_mcp.py`) that orchestrates a three-stage pipeline:

1. **Scrape** — fetch live job postings by keyword + location, save raw CSV
2. **Embed** — compute and cache HuggingFace embeddings per job description
3. **Score & Rank** — cosine similarity vs. candidate profile, LLM fit explanations,
   timestamped ranked CSV, return top-N to the cv-cover-letter-generator skill

The server uses **stdio transport** and runs with **`uv run`** — no global Python installs required.

---

## End-to-End Workflow (what the coding agent is building toward)

This is the complete user-facing flow. Every design decision below should serve this:

```
User says: "Find me 10 relevant GTM engineer jobs in Berlin and London and write CVs"
      │
      ▼
cv-cover-letter-generator skill (Claude)
      │  calls MCP tool
      ▼
scrape_jobs(keywords=["GTM engineer"], locations=["Berlin", "London"])
      │  returns: {"csv_path": "data/raw_jobs_20260315_143022.csv", "jobs_scraped": 247}
      ▼
score_jobs(csv_path="data/raw_jobs_20260315_143022.csv")
      │  computes embeddings, saves in-place
      │  returns: {"csv_path": "data/raw_jobs_20260315_143022.csv", "embedded": 247}
      ▼
get_top_jobs(csv_path=..., profile_path=".../denis_kuramshin_context_v0.2.json", top_n=10)
      │  scores, ranks, explains, saves:
      │  "output/jobs_ranked_20260315_143045.csv"
      │  returns top 10 rows as structured JSON (no embeddings)
      ▼
cv-cover-letter-generator skill
      │  iterates over top 10 jobs
      │  for each job: generates CV + cover letter using SKILL.md logic
      ▼
Writes files:
      output/Airwallex_CV.md
      output/Airwallex_cover_letter.md
      output/ListenLabs_CV.md
      output/ListenLabs_cover_letter.md
      ... (×10)
```

**File naming convention for CV outputs:**
- `output/{CompanyName}_CV.md`
- `output/{CompanyName}_cover_letter.md`
- Company name: strip spaces and special characters, e.g. "Grafana Labs" → `GrafanaLabs`

**Batching:** The ranked CSV may be 1000+ rows. The skill takes `top_n` rows (default 10).
If the user wants the next batch: call `get_top_jobs(..., top_n=20)` and skip the first 10,
or call again with an offset parameter. Keep it simple for v1 — just re-call with larger `top_n`.

---

## Reference Code

Before writing any new code, read `job_scorer.py` in this directory.
It contains working, tested implementations to reuse directly:

| Function | What it does |
|---|---|
| `clean_html(text)` | Strips HTML tags, normalises whitespace |
| `get_job_text(row)` | Extracts best text from a DataFrame row, prepends title+company |
| `profile_to_text(profile)` | Converts candidate JSON → weighted embedding text |
| `embed_texts(texts, model)` | Batch embedding, L2-normalised, progress bar |
| `scores_to_yaml(df_top, profile)` | Compact YAML serialiser for LLM consumption |

**Do not rewrite these. Import or copy them into `job_matcher_mcp.py`.**

---

## Project Structure

```
job-matcher-mcp/
├── pyproject.toml              # uv project — all deps here
├── job_matcher_mcp.py          # MCP server entrypoint
├── scraper.py                  # scraping pipeline (stub → real implementation)
├── job_scorer.py               # COPY from cv-cover-letter-generator/job_scorer.py
├── .env                        # local config (see below — OPENAI_API_KEY from Windows env)
├── tests/
│   ├── test_embedding.py       # embedding quality tests
│   ├── test_ranking.py         # pipeline correctness tests
│   └── fixtures/
│       ├── sample_jobs.csv     # 20-row labelled fixture (see Tests section)
│       └── test_profile.json   # minimal Denis profile subset for tests
├── data/                       # raw + embedded CSVs land here
│   └── .gitkeep
└── output/                     # ranked CSVs + CV markdown files land here
    └── .gitkeep
```

---

## pyproject.toml

```toml
[project]
name = "job-matcher-mcp"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "mcp[cli]>=1.0.0",
    "sentence-transformers>=3.0.0",
    "pandas>=2.0.0",
    "numpy>=1.26.0",
    "scikit-learn>=1.4.0",
    "beautifulsoup4>=4.12.0",
    "openai>=1.30.0",
    "pyyaml>=6.0",
    "python-dotenv>=1.0.0",
    "requests>=2.31.0",
]

[project.optional-dependencies]
test = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.23.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

Run server:
```bash
uv run job_matcher_mcp.py
```

Claude Desktop config (`claude_desktop_config.json`):
```json
{
  "mcpServers": {
    "job-matcher": {
      "command": "uv",
      "args": ["run", "C:/path/to/job-matcher-mcp/job_matcher_mcp.py"]
    }
  }
}
```

---

## Environment Variables (.env)

```
# OPENAI_API_KEY is set in Windows user environment variables — do NOT add here
# python-dotenv does not override existing env vars (override=False default),
# so Windows-set OPENAI_API_KEY will be picked up automatically by os.getenv()

DEFAULT_PROFILE_JSON=C:/claude_skills/custom_skills/cv-cover-letter-generator/denis_kuramshin_context_v0.2.json
DEFAULT_OUTPUT_DIR=output/
DEFAULT_DATA_DIR=data/
EMBED_MODEL=all-MiniLM-L6-v2
OPENAI_EXPLAIN_MODEL=gpt-4o-mini
EXPLAIN_BATCH_SIZE=10
DEFAULT_TOP_N=10
```

---

## Canonical CSV Schema

Derived from real job posting YAMLs (see `airwallex-director-of-software-sales.yaml`
and `listen-labs-lead-gtm-engineer.yaml` as reference).

### Raw columns produced by scraper

| Column | Type | Notes |
|---|---|---|
| `title` | str | Job title |
| `company` | str | Company name |
| `category` | str (list) | Industry categories, stored as stringified list |
| `location` | str | City, State, Country |
| `date_posted` | str | ISO date |
| `description_html` | str | Full HTML description |
| `description_text` | str | Plain text version — **primary embedding input** |
| `salary_json_min` | float | Min salary |
| `salary_json_max` | float | Max salary |
| `salary_json_currency` | str | e.g. USD |
| `salary_json_unit` | str | e.g. YEAR |
| `hiring_remote_in` | str | Remote eligible countries/regions |
| `workplace_type` | str | e.g. Remote, Hybrid, Onsite |
| `salary_range_short` | str | Display string e.g. "$120K–$160K" |
| `seniority` | str | e.g. Mid, Senior, Lead, Staff |
| `workplace_type_enum` | str | Normalised enum |
| `company_overview` | str | Short company description |
| `url` | str | Job posting URL |
| `source` | str | e.g. builtin.com, linkedin.com |
| `description` | str | Fallback description field |
| `postedDate` | str | ISO date (may duplicate date_posted) |
| `salary` | str | Display string |
| `id` | str/int | Source ID |
| `is_gtm_technical` | bool | Classifier flag |
| `tech_stack` | str (list) | Technologies mentioned, stringified list |
| `min_years_exp` | int | Minimum years experience required |
| `salary_min` | float | May duplicate salary_json_min |
| `salary_max` | float | May duplicate salary_json_max |
| `reason` | str | Classifier rationale |

### Normalised columns (added by `normalise_columns()`)

| Raw name | Normalised name | Note |
|---|---|---|
| `salary_json_min` | `salary_min` | keep raw too if both present |
| `salary_json_max` | `salary_max` | |
| `salary_json_currency` | `salary_currency` | |
| `workplace_type_enum` | `workplace_type` | |
| *(new)* | `embedding` | JSON string of float list |
| *(new)* | `embedding_model` | model name used, e.g. `all-MiniLM-L6-v2` |
| *(new)* | `similarity_score` | float 0.0–1.0 |
| *(new)* | `fit_explanation` | str, 2–3 sentences from gpt-4o-mini |

`normalise_columns(df)` must:
1. Rename columns per map above
2. Strip leading/trailing whitespace from all column names
3. Replace any remaining spaces in column names with underscores
4. Lowercase all column names
5. Return the modified df

---

## Tool 1 — `scrape_jobs` (Stub → Real)

### Purpose
Fetch live job postings matching keywords × locations. Save raw CSV.
This is a **stub in v1** — implement the interface cleanly so the real scraper
(existing pipeline) can be plugged in by adapting `scraper.py`.

### Signature
```python
@mcp.tool()
async def scrape_jobs(
    keywords: list[str] | None = None,    # default: DEFAULT_KEYWORDS constant
    locations: list[str] | None = None,   # default: DEFAULT_LOCATIONS constant
    max_results_per_query: int = 100,
    output_path: str | None = None,       # default: data/raw_jobs_YYYYMMDD_HHMMSS.csv
) -> dict:
    """
    Scrape job postings matching keywords × locations.
    Returns: {
        "status": "ok",
        "csv_path": "data/raw_jobs_20260315_143022.csv",
        "jobs_scraped": 247,
        "queries_run": 6
    }
    """
```

### Default constants (module level, easily changed)
```python
DEFAULT_KEYWORDS = ["GTM engineer"]
DEFAULT_LOCATIONS = ["Berlin", "London", "New York", "San Francisco", "Boston", "US remote"]
```

### Stub implementation for v1
```python
async def scrape_jobs(keywords=None, locations=None, max_results_per_query=100, output_path=None):
    keywords = keywords or DEFAULT_KEYWORDS
    locations = locations or DEFAULT_LOCATIONS

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = output_path or f"data/raw_jobs_{timestamp}.csv"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    # --- STUB: replace this block with real scraper.py call ---
    # Real implementation should:
    #   1. For each (keyword, location) pair, call the scraping pipeline
    #   2. Collect results into a single DataFrame
    #   3. Deduplicate on (title, company, url)
    #   4. Ensure all canonical schema columns are present (fill missing with empty string)
    #   5. Call normalise_columns(df)
    #   6. Save to out_path
    #
    # Scraper interface to implement in scraper.py:
    #   def scrape(keywords: list[str], locations: list[str],
    #              max_per_query: int) -> pd.DataFrame
    #
    # For now, raise NotImplementedError with helpful message:
    raise NotImplementedError(
        "scrape_jobs stub: implement scraper.py with def scrape(keywords, locations, max_per_query) -> pd.DataFrame. "
        "See TASK.md §Tool 1 for the expected column schema."
    )
    # --- END STUB ---

    df = normalise_columns(df)
    df.to_csv(out_path, index=False)
    logging.info(f"Scraped {len(df)} jobs → {out_path}")
    return {"status": "ok", "csv_path": out_path, "jobs_scraped": len(df), "queries_run": len(keywords) * len(locations)}
```

### scraper.py interface contract
The coding agent must create `scraper.py` with at minimum:
```python
def scrape(
    keywords: list[str],
    locations: list[str],
    max_per_query: int = 100,
) -> pd.DataFrame:
    """
    Returns a DataFrame with the canonical schema columns.
    Missing columns should be filled with empty string or None.
    Deduplication on (title, company, url) should be applied before returning.
    """
    ...
```

Denis will adapt his existing scraping code into this function.

---

## Tool 2 — `score_jobs`

### Purpose
Load raw CSV, compute HuggingFace embeddings per job, store in `embedding` column, save in-place.
Expensive step — cached so re-runs skip already-embedded rows.

### Signature
```python
@mcp.tool()
async def score_jobs(
    csv_path: str,
    embed_model: str | None = None,   # default from EMBED_MODEL env var
    force_reembed: bool = False,
) -> dict:
    """
    Compute and cache job embeddings.
    Returns: {
        "status": "ok",
        "csv_path": "data/raw_jobs_20260315_143022.csv",
        "embedded": 230,
        "skipped_cached": 17,
        "model_used": "all-MiniLM-L6-v2"
    }
    """
```

### Implementation notes
- Use `get_job_text(row)` from `job_scorer.py` for text extraction
- **Cache logic**: skip rows where `embedding` is non-null AND `embedding_model` matches
  current model, unless `force_reembed=True`
- Store embedding as `json.dumps(vector.tolist())`
- Store model name in `embedding_model` column
- Save CSV back to same path (in-place)
- Model singleton: loaded once at server startup, never per-call

---

## Tool 3 — `get_top_jobs`

### Purpose
Cosine similarity scoring, ranking, LLM explanations, timestamped output CSV, return top-N.

### Signature
```python
@mcp.tool()
async def get_top_jobs(
    csv_path: str,
    profile_path: str | None = None,   # default: DEFAULT_PROFILE_JSON
    top_n: int | None = None,           # None = all rows; default from DEFAULT_TOP_N env
    min_score: float = 0.0,
    explain: bool = True,
    output_dir: str | None = None,      # default: DEFAULT_OUTPUT_DIR
) -> dict:
    """
    Score, rank, explain, save, return top matches.
    Returns: {
        "status": "ok",
        "output_csv": "output/jobs_ranked_20260315_143045.csv",
        "total_scored": 247,
        "returned": 10,
        "score_range": {"max": 0.847, "min": 0.612, "median": 0.703},
        "top_jobs": [
            {
                "title": "GTM Engineer",
                "company": "Tapcheck",
                "location": "USA",
                "salary": "USD 160,000–190,000",
                "seniority": "Mid",
                "workplace_type": "Remote",
                "tech_stack": "['Python', 'SQL', 'Clay', 'n8n']",
                "min_years_exp": "3",
                "url": "https://...",
                "similarity_score": 0.847,
                "fit_explanation": "Strong match: Denis's agentic pipeline project directly..."
            },
            ...
        ]
    }
    """
```

### Step-by-step implementation

**Step 1 — Load and validate**
```python
df = pd.read_csv(csv_path)
df = normalise_columns(df)
if "embedding" not in df.columns or df["embedding"].isna().all():
    raise ValueError("No embeddings found. Run score_jobs() first.")
profile = json.loads(Path(profile_path).read_text(encoding="utf-8"))
```

**Step 2 — Profile master vector**
```python
profile_text = profile_to_text(profile)   # from job_scorer.py
profile_vec = _model.encode([profile_text], normalize_embeddings=True)[0]
```

**Step 3 — Cosine similarity (vectorised)**
```python
valid = df["embedding"].notna()
job_vecs = np.array([json.loads(e) for e in df.loc[valid, "embedding"]])
scores = cosine_similarity([profile_vec], job_vecs)[0]
df.loc[valid, "similarity_score"] = scores
```

**Step 4 — Sort and filter**
```python
df = df.sort_values("similarity_score", ascending=False).reset_index(drop=True)
if min_score > 0:
    df = df[df["similarity_score"] >= min_score]
effective_top_n = top_n or int(os.getenv("DEFAULT_TOP_N", 10))
df_top = df.head(effective_top_n)
```

**Step 5 — Fit explanations via gpt-4o-mini**

Only for rows where `fit_explanation` is null. Batch with `asyncio.gather`.

```python
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
```

`candidate_summary` = `profile["summary"]["elevator_pitch"]`
`description_snippet` = first 600 chars of `description_text`

**Step 6 — Save ranked CSV**
```python
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
out_path = Path(output_dir) / f"jobs_ranked_{timestamp}.csv"
df_out = df_top.drop(columns=["embedding", "embedding_model"], errors="ignore")
df_out.to_csv(out_path, index=False)
```

**Step 7 — Return structured JSON**
Return these columns only (no embeddings):
`title, company, location, salary, seniority, workplace_type, tech_stack,
min_years_exp, url, similarity_score, fit_explanation`

Include `score_range` stats for the skill to use in its rationale.

---

## Server Startup

```python
import os, sys, json, logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from sentence_transformers import SentenceTransformer

load_dotenv()  # loads .env; does NOT override Windows env vars

# Logging to stderr only — stdout is reserved for MCP protocol
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

mcp = FastMCP("job-matcher")

# Load embedding model once — ~80MB download on first run, cached by HuggingFace
logging.info("Loading embedding model...")
_model = SentenceTransformer(os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2"))
logging.info("Embedding model ready.")

# ... tool definitions ...

if __name__ == "__main__":
    mcp.run(transport="stdio")
```

---

## Column Name Convention

`normalise_columns(df)` utility — call immediately after every `pd.read_csv()`:

```python
COLUMN_RENAME_MAP = {
    "salary_json_min": "salary_min",
    "salary_json_max": "salary_max",
    "salary_json_currency": "salary_currency",
    "salary_json_unit": "salary_unit",
    "workplace_type_enum": "workplace_type",
}

def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Strip whitespace from column names
    df.columns = df.columns.str.strip()
    # Lowercase + replace spaces with underscores
    df.columns = df.columns.str.lower().str.replace(" ", "_", regex=False)
    # Apply explicit rename map
    df = df.rename(columns=COLUMN_RENAME_MAP)
    return df
```

---

## Tests

### Test philosophy
1. **Quality tests** — do embeddings rank relevant jobs above irrelevant ones?
2. **Correctness tests** — does the pipeline produce the right shape, columns, ordering?

### `tests/fixtures/sample_jobs.csv`

20 rows, hand-labelled. Required columns:
`title, company, location, description_text, tech_stack, salary, seniority, expected_rank_tier`

Label distribution:
- `"top"` (5 rows): GTM Engineer, ML Engineer, Data Scientist, Solutions Engineer, LLM Engineer
- `"bottom"` (5 rows): Civil Engineer, Registered Nurse, Warehouse Supervisor, HR Manager, Accountant
- `"mid"` (10 rows): Sales Engineer, Product Manager, Data Analyst, DevOps Engineer, etc.

### `tests/fixtures/test_profile.json`

Minimal subset of `denis_kuramshin_context_v0.2.json` — just enough for `profile_to_text()` to work:
`summary`, `technical_skills`, `ats_keywords_pool`, `open_roles_targeted`, `experience` (2 roles max).

### `tests/test_embedding.py` — Quality

```python
def test_top_jobs_outscore_bottom():
    """Top-tier jobs must score meaningfully higher than bottom-tier."""
    top_mean = ...   # mean similarity_score of rows with expected_rank_tier == "top"
    bottom_mean = ...
    assert top_mean > bottom_mean + 0.05

def test_top_jobs_in_top10():
    """At least 4/5 top-tier jobs must appear in the top-10 ranked results."""
    ...

def test_score_range():
    """All scores must be in [0, 1] with no NaN."""
    assert 0.0 <= scores.min() and scores.max() <= 1.0
    assert not np.isnan(scores).any()

def test_profile_to_text_nonempty():
    """profile_to_text must return >200 chars containing key skills."""
    text = profile_to_text(load_test_profile())
    assert len(text) > 200
    assert any(kw in text.lower() for kw in ["python", "machine learning", "llm", "gcp"])
```

### `tests/test_ranking.py` — Correctness

```python
def test_normalise_columns():
    df = pd.DataFrame(columns=["salary_json_min", "workplace_type_enum", "description_text"])
    df = normalise_columns(df)
    assert "salary_min" in df.columns
    assert "workplace_type" in df.columns

def test_embedding_cache_skips_existing():
    """Rows with existing embeddings from the same model must not be re-embedded."""
    ...

def test_output_has_no_embedding_column():
    """Ranked CSV output must not contain the embedding column."""
    df_out = pd.read_csv(ranked_csv_path)
    assert "embedding" not in df_out.columns

def test_output_sorted_descending():
    """Output CSV must be sorted by similarity_score descending."""
    scores = pd.read_csv(ranked_csv_path)["similarity_score"].tolist()
    assert scores == sorted(scores, reverse=True)

def test_fit_explanations_populated_for_top5():
    """Top 5 rows must have non-empty fit_explanation strings."""
    df = pd.read_csv(ranked_csv_path).head(5)
    assert all(isinstance(r, str) and len(r) > 20 for r in df["fit_explanation"])

def test_model_name_stored_in_csv():
    """Each embedded row must record which model produced the embedding."""
    df = pd.read_csv(embedded_csv_path)
    assert "embedding_model" in df.columns
    assert df["embedding_model"].notna().all()
```

Run all:
```bash
uv run pytest tests/ -v
```

---

## Suggested Build Order

1. `pyproject.toml` → `uv sync` to verify deps
2. Copy `job_scorer.py` from `../cv-cover-letter-generator/`
3. Create `tests/fixtures/sample_jobs.csv` (20 rows, hand-labelled) + `test_profile.json`
4. Write `normalise_columns()` + `tests/test_ranking.py` — get passing
5. Write `tests/test_embedding.py` — run against `job_scorer.py` directly, get passing
6. Build `job_matcher_mcp.py` scaffold: server startup + model load + logging
7. Implement `score_jobs()` + its cache logic
8. Implement `get_top_jobs()` including OpenAI explanation batching
9. Implement `scrape_jobs()` stub with `NotImplementedError` + clean interface
10. Manual end-to-end test via MCP Inspector: `uv run job_matcher_mcp.py`
11. Denis adapts existing scraper → `scraper.py` with `scrape()` function signature
12. Replace stub with real `scraper.py` call, test full pipeline

---

## Notes for Coding Agent

- **No `print()` inside tools** — stdout is the MCP protocol wire. Use `logging` to stderr only.
- **No `asyncio.run()` inside tools** — FastMCP owns the event loop. All tools are `async def`.
- **OpenAI**: use `AsyncOpenAI`, batch explanation calls with `asyncio.gather`.
- **Embedding model**: module-level singleton `_model`. Load once at startup. Never per-call.
- **Embeddings in CSV**: serialised as `json.dumps(list)`. Deserialise with `json.loads()` before numpy ops.
- **Model consistency**: job embeddings and profile vector MUST use the same model.
  If `EMBED_MODEL` changes, call `score_jobs(force_reembed=True)` to invalidate cache.
  The `embedding_model` column enables detection of stale embeddings.
- **`normalise_columns()`**: call immediately after every `pd.read_csv()`, before anything else.
- **Fit explanations**: only generate for `top_n` rows, not the full CSV. Cache in column.
- **CV output files**: the cv-cover-letter-generator skill (not this MCP server) writes the
  `.md` files. The MCP server only returns structured JSON. The skill handles all CV logic.
- **Windows env vars**: `OPENAI_API_KEY` is in Windows user environment — `os.getenv()` finds it
  automatically. `python-dotenv` does not override existing env vars by default. Do not add
  `OPENAI_API_KEY` to `.env`.

---

## README Template (for coding agent to populate)

When the build is complete, create `README.md` with these sections:

```
# Job Matcher MCP Server

## What it does
## Requirements
## Installation (uv)
## Configuration (.env)
## Usage
### Via Claude Desktop
### Via MCP Inspector (manual testing)
## Tools reference
### scrape_jobs
### score_jobs
### get_top_jobs
## Full workflow example
## Adapting scraper.py
## Swapping the embedding model
## Running tests
## File outputs reference
```
