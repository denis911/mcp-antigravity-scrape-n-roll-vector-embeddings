# EXPAND.md — Generalisation & Improvement Spec

## Purpose

This document is a combined **code review** and **expansion task list** for a coding agent.
The current implementation works for GTM engineer roles but has hardcoded assumptions
that prevent it from working well across other job types (sales, channel, biotech, data science,
part-time, European markets, etc.). This document explains every issue found and exactly
what to change.

Read all three source files before starting:
- `scraper.py` — scraping + LLM classification pipeline
- `job_matcher_mcp.py` — MCP server, three tools
- `job_scorer.py` — embedding utilities (mostly fine, minor issues noted)

---

## Summary of Issues Found

| # | File | Severity | Issue |
|---|---|---|---|
| 1 | `scraper.py` | 🔴 Critical | `EXTRACTION_PROMPT` is hardcoded for GTM roles — filters out sales, biotech, channel jobs |
| 2 | `scraper.py` | 🔴 Critical | BuiltIn URL builder is US-centric — European locations silently return no results |
| 3 | `scraper.py` | 🟡 Medium | `is_gtm_technical` field name is misleading when classifying non-GTM roles |
| 4 | `scraper.py` | 🟡 Medium | Duplicate column issue (`workplace_type` / `workplace_type.1`) not fixed before saving |
| 5 | `scraper.py` | 🟡 Medium | Sync `OpenAI` client used inside what may be an async context |
| 6 | `job_matcher_mcp.py` | 🟡 Medium | `scrape_jobs` MCP tool timeout on multiple keyword × location queries |
| 7 | `job_matcher_mcp.py` | 🟡 Medium | `get_top_jobs` does not filter interns or apply seniority/salary floor |
| 8 | `job_matcher_mcp.py` | 🟡 Medium | `embed_texts()` uses `print()` — corrupts stdio MCP protocol |
| 9 | `job_matcher_mcp.py` | 🟢 Minor | `workplace_type` NaN in output due to duplicate column handling |
| 10 | `job_scorer.py` | 🟢 Minor | `embed_texts()` uses `print()` not `logging` — fine for CLI, bad when imported by MCP |
| 11 | `job_matcher_mcp.py` | 🟢 Minor | No `min_seniority` or `employment_type` filter exposed as tool parameters |

---

## Issue 1 — CRITICAL: Extraction prompt hardcoded for GTM

### Current code (`scraper.py` lines ~50-85)

```python
EXTRACTION_PROMPT = """
You are a GTM (Go-To-Market) Engineering expert. Analyze the following job descriptions...

### OBJECTIVE:
1. **Relevancy Filter**: Identify if the role is a "Technical GTM" role
   - Skip pure non-technical Sales/Marketing roles (e.g., BDR, Account Executive, Content Manager).
"""
```

### Problem
The prompt explicitly tells the LLM to mark `is_gtm_technical=False` for Account Executive,
Sales Director, Channel Manager, Bioinformatics Scientist, Data Engineer, etc.
Any role outside the GTM engineering niche gets filtered out. This is the root cause of why
sales and biotech roles score False and get buried.

### Fix — make the prompt and field name dynamic based on a `job_domain` parameter

**Step 1**: Add `job_domain` parameter to `extract_structured_data()` and `scrape()`:

```python
def scrape(
    keywords: list[str],
    locations: list[str],
    max_per_query: int = 100,
    job_domain: str = "any",          # NEW: "gtm", "sales", "biotech", "data", "any"
) -> pd.DataFrame:
```

**Step 2**: Replace the hardcoded prompt with a dynamic one based on `job_domain`:

```python
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
Return a JSON array, one object per job, same order as input:
[
  {{
    "id": "original_id",
    "is_relevant": true,
    "tech_stack": ["tool1"],
    "seniority": "Senior",
    "employment_type": "Full-time",
    "min_years_exp": 5,
    "salary_min": 150000,
    "salary_max": 200000,
    "reason": "..."
  }}
]
"""
```

**Step 3**: Replace `is_gtm_technical` field with `is_relevant` everywhere.
- In `scraper.py`: rename in canonical_columns list and in all references
- In `job_matcher_mcp.py`: update any references to `is_gtm_technical`
- The `is_relevant` field is now domain-neutral

**Step 4**: Add `employment_type` to canonical columns list in `scrape()`.

**Step 5**: Pass `job_domain` through from `scrape_jobs` MCP tool:

```python
@mcp.tool()
async def scrape_jobs(
    keywords: list[str] | None = None,
    locations: list[str] | None = None,
    max_results_per_query: int = 100,
    job_domain: str = "any",           # NEW — "gtm", "sales", "biotech", "data", "any"
    output_path: str | None = None,
) -> dict:
```

---

## Issue 2 — CRITICAL: BuiltIn URL builder fails for European locations

### Current code (`scraper.py`)

```python
start_urls.append(f"https://builtin.com/jobs?search={kw_url}&location={loc_url}")
```

### Problem
BuiltIn is a US job board. Searching `location=Berlin` or `location=London` returns zero
or near-zero results because BuiltIn doesn't index European jobs. This is why the London/Berlin
scrape returned mostly US jobs during testing.

### Fix — multi-source scraper with source routing per location

**Step 1**: Add a `SOURCE_MAP` that routes locations to appropriate job sources:

```python
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
        "fallback1": "scrapio/linkedin-search-jobs-scraper",
    },
}
```

**Step 2**: Route URLs based on location in `scrape()`:

```python
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
```

**Step 3**: Call the right Apify actor per source in `scrape_apify()`:

```python
def scrape_apify(urls_by_source: dict[str, list[str]], max_items: int) -> list[dict]:
    all_results = []
    for source, urls in urls_by_source.items():
        if not urls:
            continue
        actors = APIFY_ACTORS.get(source, APIFY_ACTORS["builtin"])
        for url in urls:
            # ... call actor as before, with primary/fallback logic ...
    return all_results
```

**Note for coding agent**: Verify the correct LinkedIn Apify actor ID before deploying.
Common options: `curious_coder/linkedin-jobs-scraper`, `bebity/linkedin-jobs-scraper`.
Test with 2-3 URLs first to confirm output schema matches BuiltIn schema.
I have downloaded documentation for 

---

## Issue 3 — MEDIUM: `is_gtm_technical` field name is misleading

### Fix
Already covered in Issue 1 — rename to `is_relevant` everywhere.
Update `canonical_columns` list in `scraper.py`, update `COLUMN_RENAME_MAP` in
`job_matcher_mcp.py` if needed, update any filtering logic that references the old name.

---

## Issue 4 — MEDIUM: Duplicate column names in output CSV

### Current problem
When both Apify actor outputs and field normalisation produce `workplace_type`, pandas
creates `workplace_type` and `workplace_type.1`. This causes `workplace_type` to show
as NaN in MCP tool output.

### Fix — add deduplication step in `normalise_columns()`

```python
def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    # ... existing steps ...

    # NEW: Remove duplicate columns (keep first occurrence)
    df = df.loc[:, ~df.columns.duplicated(keep="first")]

    return df
```

Also fix in `scraper.py` `scrape_apify()` — avoid creating the duplicate in the first place
by not setting `workType` if `workplace_type` already exists:

```python
# Before:
if not item.get("workType") and item.get("workplace_type"):
    item["workType"] = item["workplace_type"]

# After: just map workplace_type into the canonical name directly
if not item.get("workplace_type") and item.get("workType"):
    item["workplace_type"] = item["workType"]
```

---

## Issue 5 — MEDIUM: Sync OpenAI client in potentially async context

### Current code (`scraper.py`)
```python
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))  # sync client
```

The `scraper.py` `extract_structured_data()` is called from `scrape_jobs()` MCP tool
which is `async def`. The sync OpenAI client blocks the event loop during batch processing.

### Fix
Use `run_in_executor` to offload blocking calls, OR switch to `AsyncOpenAI` + `asyncio.gather`
for parallel batch processing (same pattern already used in `_explain_job` in `job_matcher_mcp.py`).

For simplicity in v1, wrap in executor:
```python
import asyncio

async def extract_structured_data_async(raw_data, model="gpt-4o-mini", batch_size=20):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        extract_structured_data,  # existing sync function
        raw_data, model, batch_size
    )
```

Then in `scrape_jobs` MCP tool, `await extract_structured_data_async(raw_results)`.

---

## Issue 6 — MEDIUM: MCP tool timeout on large keyword × location matrix

### Problem
1 keyword × 2 locations = 2 Apify queries + OpenAI classification ≈ 60-90 seconds.
3 keywords × 6 locations = 18 Apify queries ≈ 10-15 minutes → MCP timeout.

### Fix — two changes

**A. Add `timeout` config to FastMCP server** (if supported by your MCP version):
```python
mcp = FastMCP("job-matcher", timeout=300)  # 5 minute timeout
```

**B. Process Apify queries in parallel** using `asyncio.gather` inside `scrape_apify()`:

```python
async def scrape_apify_async(urls_by_source: dict, max_items: int) -> list[dict]:
    """Parallel Apify scraping — runs all URL queries concurrently."""
    import asyncio
    loop = asyncio.get_event_loop()

    async def fetch_one(source: str, url: str) -> list[dict]:
        actors = APIFY_ACTORS.get(source, APIFY_ACTORS["builtin"])
        return await loop.run_in_executor(None, _scrape_single_url, url, actors, max_items)

    tasks = [
        fetch_one(source, url)
        for source, urls in urls_by_source.items()
        for url in urls
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    all_items = []
    for r in results:
        if isinstance(r, Exception):
            logging.error(f"Scrape task failed: {r}")
        else:
            all_items.extend(r)
    return all_items
```

**C. Add a `max_queries` safety cap** as a parameter:
```python
async def scrape_jobs(
    ...
    max_total_queries: int = 6,   # NEW safety cap — prevents runaway timeouts
) -> dict:
    # Warn if keywords × locations exceeds cap
    total = len(keywords) * len(locations)
    if total > max_total_queries:
        logging.warning(f"Query count {total} exceeds cap {max_total_queries}. Truncating.")
        # Take first max_total_queries combinations
```

---

## Issue 7 — MEDIUM: No pre-filtering for seniority, employment type, or salary floor

### Problem
The intern LogicGate role ranked #1 in scoring because cosine similarity doesn't
understand that a $40/hr intern role is irrelevant for a 20-year enterprise professional.
The current `get_top_jobs` has only `min_score` filtering.

### Fix — add pre-filter parameters to `get_top_jobs`

```python
@mcp.tool()
async def get_top_jobs(
    csv_path: str,
    profile_path: str | None = None,
    top_n: int | None = None,
    min_score: float = 0.0,
    explain: bool = True,
    output_dir: str | None = None,
    # NEW FILTERS:
    exclude_seniority: list[str] | None = None,   # e.g. ["Intern", "Junior"]
    employment_types: list[str] | None = None,    # e.g. ["Full-time"] — None = all
    min_salary: float | None = None,              # e.g. 80000 — filter below this
    relevant_only: bool = True,                   # filter is_relevant == True
) -> dict:
```

Implementation — apply filters before embedding/scoring:
```python
# Apply pre-filters
if relevant_only and "is_relevant" in df.columns:
    df = df[df["is_relevant"].astype(str).str.lower() == "true"]

if exclude_seniority:
    excl = [s.lower() for s in exclude_seniority]
    df = df[~df["seniority"].str.lower().isin(excl)]

if employment_types:
    emp = [e.lower() for e in employment_types]
    df = df[df["employment_type"].str.lower().isin(emp)]

if min_salary and "salary_min" in df.columns:
    df = df[pd.to_numeric(df["salary_min"], errors="coerce").fillna(0) >= min_salary]

logging.info(f"After pre-filters: {len(df)} rows remain from {original_count}")
```

---

## Issue 8 & 10 — MEDIUM/MINOR: `print()` in `embed_texts()` corrupts MCP protocol

### Current code (`job_scorer.py`)
```python
def embed_texts(texts, model, batch_size=64):
    print(f"  Embedding {len(texts)} texts in batches of {batch_size}...")  # ← BAD
    embeddings = model.encode(..., show_progress_bar=True, ...)             # ← also prints
```

Any `print()` or progress bar output to stdout corrupts the MCP stdio protocol.
`show_progress_bar=True` writes to stdout by default.

### Fix
```python
def embed_texts(texts, model, batch_size=64):
    logging.info(f"Embedding {len(texts)} texts in batches of {batch_size}...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,   # ← MUST be False when called from MCP server
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embeddings
```

Note: `job_scorer.py` is also used as a standalone CLI tool where progress bars are useful.
Add a `verbose: bool = False` parameter so the CLI can pass `verbose=True` and the MCP
server always passes `verbose=False`.

---

## Issue 9 — MINOR: `workplace_type` NaN in MCP output

Already fixed by Issue 4 (duplicate column deduplication). No additional change needed.

---

## Issue 11 — MINOR: No `employment_type` or `min_salary` exposed as MCP parameters

Already fixed by Issue 7 above.

---

## New Feature: `list_saved_csvs` utility tool

Add a fourth lightweight MCP tool for convenience — lets the skill (or user) see what
cached CSVs are available without needing filesystem access:

```python
@mcp.tool()
async def list_saved_csvs(
    data_dir: str | None = None,
    output_dir: str | None = None,
) -> dict:
    """
    List available raw and ranked CSVs with row counts and timestamps.
    Useful for re-running score_jobs or get_top_jobs on a previous scrape
    without re-scraping.
    Returns: {
        "raw_csvs": [{"path": "...", "rows": N, "created": "..."}],
        "ranked_csvs": [{"path": "...", "rows": N, "created": "..."}]
    }
    """
    data_dir = data_dir or os.getenv("DEFAULT_DATA_DIR", "data/")
    output_dir = output_dir or os.getenv("DEFAULT_OUTPUT_DIR", "output/")

    def summarise_dir(d: str) -> list[dict]:
        results = []
        for p in sorted(Path(d).glob("*.csv")):
            try:
                df = pd.read_csv(p, nrows=0)  # headers only for row count
                row_count = sum(1 for _ in open(p)) - 1
                results.append({
                    "path": str(p),
                    "rows": row_count,
                    "created": datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
                })
            except Exception:
                pass
        return results

    return {
        "raw_csvs": summarise_dir(data_dir),
        "ranked_csvs": summarise_dir(output_dir),
    }
```

---

## New Feature: `job_domain` auto-detection from keywords

Instead of requiring the user to specify `job_domain`, auto-detect it from keywords:

```python
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
```

Use in `scrape_jobs` when `job_domain` is not explicitly passed:
```python
job_domain = job_domain or detect_domain(keywords)
logging.info(f"Auto-detected job_domain: {job_domain}")
```

---

## Updated `.env` additions

```
# New in EXPAND
DEFAULT_JOB_DOMAIN=any
DEFAULT_EXCLUDE_SENIORITY=Intern
DEFAULT_EMPLOYMENT_TYPE=Full-time
DEFAULT_MIN_SALARY=0
MAX_TOTAL_QUERIES=8
```

---

## Updated canonical columns list

Add `employment_type` and rename `is_gtm_technical` → `is_relevant`:

```python
canonical_columns = [
    "title", "company", "category", "location", "date_posted",
    "description_html", "description_text",
    "salary_json_min", "salary_json_max", "salary_json_currency", "salary_json_unit",
    "hiring_remote_in", "workplace_type", "salary_range_short",
    "seniority", "employment_type",                  # employment_type NEW
    "company_overview", "url", "source",
    "description", "postedDate", "salary", "id",
    "is_relevant",                                   # renamed from is_gtm_technical
    "tech_stack", "min_years_exp", "salary_min", "salary_max", "reason"
]
```

## Make sure we can read older scraped GTM eng file too

Add an option to read, calculate embeddings and rank older scraped file - 
"C:\claude_skills\MCP\mcp-antigravity-scrape-n-roll-vector-embeddings\data\apify_raw_export.csv"
-- we would need it to experiment with the quality of embeddings and ranking...
So instead of scraping 800+ jobs from Builtin we may already re-use it.
It may require to ignore irrelevant columns in old files though.

---

## Build Order for Coding Agent

1. **Fix Issue 8/10 first** — `print()` → `logging` in `embed_texts()`, `show_progress_bar=False`.
   This is a one-line fix that unblocks clean MCP operation.

2. **Fix Issue 4** — add `df.loc[:, ~df.columns.duplicated()]` to `normalise_columns()`.
   Quick fix, eliminates NaN in workplace_type.

3. **Fix Issue 1** — refactor `EXTRACTION_PROMPT` into `DOMAIN_PROMPTS` dict + template.
   Rename `is_gtm_technical` → `is_relevant`. Add `employment_type` to extraction.
   Add `job_domain` parameter to `scrape()` and `scrape_jobs()`.
   Add `detect_domain()` auto-detection.

4. **Fix Issue 7** — add pre-filter parameters to `get_top_jobs()`.

5. **Fix Issue 2** — add `SOURCE_MAP` + multi-source URL builder + LinkedIn actor routing.
   Test LinkedIn actor separately before integrating.

6. **Fix Issues 5 & 6** — async OpenAI in scraper + parallel Apify queries + timeout config.

7. **Add `list_saved_csvs` tool** — small, independent, useful for debugging.

8. **Update `.env`** with new default values.

9. **Update tests** — add test cases for:
   - `detect_domain()` returns correct domain for each keyword set
   - `normalise_columns()` deduplicates columns
   - `get_top_jobs()` correctly filters out interns when `exclude_seniority=["Intern"]`
   - `scrape()` called with `job_domain="sales"` does not filter out Account Executive roles

10. **Update README** to reflect new parameters and multi-domain capability.

11. **Double-check** if any unresolved issues or new features are still not done.
If all good - create ver2.md report, so I can test it in real workflow.
