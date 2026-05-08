# Job Matcher MCP Server

STDIO-based MCP server to collect, extract, and semantically rank job postings data, intended to be used with LLM agents and skills while saving tokens.

## What it does
Builds a local MCP server (`job_matcher_mcp.py`) that orchestrates a smart pipeline:
1. **Scrape & Extract** — fetches jobs by keyword + location and uses an LLM to extract structured data.
   - **Modular Architecture**: Uses dedicated scraper modules (`builtin_scraper.py` for BuiltIn/Apify, `serper_scraper.py` for Google ATS footprints, `japan_scraper.py` for Japanese job boards, and `linkedin_scraper.py` for LinkedIn via Apify). Extraction logic is centralized in `extractor.py`.
   - **Hybrid Routing**: Automatically routes US locations to BuiltIn (via Apify), EU/Global locations to LinkedIn (via Apify) or Google Serper (ATS footprints), and Japanese locations (e.g. "Japan", "Tokyo") to Japanese-specific job boards within the same run.
2. **A/B Test Strategies** — allows comparing different keyword sets side-by-side to find the highest-quality yield via `compare_searches()`.
3. **Embed** — computes and caches local HuggingFace embeddings for the descriptions (runs locally, completely free).
4. **Filter & Score** — applies active pre-filters (salary floors, seniority exclusion, relevancy) before ranking by cosine similarity against your profile.

## Requirements
- Python >= 3.11
- `uv` package manager
- `OPENAI_API_KEY` (required for structured extraction and fit explanation)
- `APIFY_API_TOKEN` (if using Apify backend)
- `SERPER_API_KEY` (if using Serper backend)

## Installation (uv)

(If uv is not installed use choco (choco install uv).)

```bash
uv sync --extra test
```

## Configuration (.env)

The server uses a `.env` file for defaults. API keys should be in your Windows Environment Variables.

```powershell
# Verify keys are set
echo $env:OPENAI_API_KEY
echo $env:APIFY_API_TOKEN
echo $env:SERPER_API_KEY
```

Available `.env` settings:
- `SCRAPER_BACKEND`: *(Deprecated)* Location routing is now handled automatically via a hybrid approach.
- `DEFAULT_JOB_DOMAIN`: `any`, `gtm`, `sales`, `biotech`, `data`.
- `DEFAULT_MIN_SALARY`: Filter out jobs below this base salary (e.g. `80000`).
- `DEFAULT_EXCLUDE_SENIORITY`: Comma-separated list (e.g. `Intern,Junior`).
- `MAX_TOTAL_QUERIES`: Safety cap for parallel scraping (default: 8).

## Tools

- **`scrape_jobs`**: Parallel scraping with domain auto-detection and multi-source routing.
- **`compare_searches`**: Run A/B tests across multiple keyword strategies to find the highest-quality yield.
- **`snipe_url`**: Fetch, parse, embed, and score any single job URL instantly (~$0.002).
- **`score_jobs`**: Computes local HuggingFace embeddings for a CSV.
- **`get_top_jobs`**: Ranks jobs against your `test_profile.json` applying configurable pre-filters.
- **`list_saved_csvs`**: Utility to view all cached CSV files in `data/` and `output/`.

## Keyword Strategy

When scraping jobs, use these expanded taxonomy categories to maximize search yield for technical/commercial hybrid roles:

### Category A — Pre-sales / Solutions Engineering (HIGH PRIORITY)
Focuses on roles that bridge technical depth with commercial outcomes.
- **Keywords**: `pre-sales engineer`, `solutions engineer`, `solutions consultant`, `sales engineer`, `forward deployed engineer`, `applied AI engineer`, `technical sales engineer`, `AI solutions consultant`.
- **Rationale**: Highly qualified for candidates with dual technical/commercial backgrounds. duvo.ai (4.80/5.0) came from this category.

### Category B — Technology-based Search
Searching by stack/domain rather than job function.
- **Keywords**: `AI automation sales`, `LLM solutions engineer`, `agentic AI sales`, `GCP partner sales`, `PostgreSQL solutions`, `data infrastructure sales`, `AI consulting presales`.
- **Rationale**: Catches high-relevancy roles at niche startups that may use non-standard titles.

### Category C — Customer Success / TAM (MEDIUM PRIORITY)
Post-sales technical relationship management.
- **Keywords**: `technical account manager AI`, `customer success manager AI SaaS`, `AI platform customer success`, `enterprise customer success data`, `technical customer success`.
- **Rationale**: Lower quota pressure, higher relationship depth, but still commercially relevant.

### Category D — FDE / Applied Engineering (STRETCH)
Coding-heavy implementation roles at AI startups.
- **Keywords**: `forward deployed engineer AI`, `applied AI engineer`, `implementation engineer AI SaaS`, `solutions architect AI`, `AI implementation consultant`.
- **Rationale**: Feasible for candidates with prototyping track records and cloud expertise. Line between PSE and FDE is often thin at early startups.

## Usage

### A/B Testing Example

To evaluate which strategy finds better-fitting roles, use `compare_searches()`:

```python
compare_searches(
    search_strategies=[
        {
            "name": "traditional_sales",
            "keywords": ["account executive", "founding AE"],
            "locations": ["Remote", "London"],
            "job_domain": "sales"
        },
        {
            "name": "presales_se",
            "keywords": ["pre-sales engineer", "solutions engineer"],
            "locations": ["Remote", "London"],
            "job_domain": "sales"
        }
    ],
    max_results_per_query=10
)
```

### Via Claude Desktop / Cursor
Add to your config:
```json
{
  "mcpServers": {
    "job-matcher": {
      "command": "uv",
      "args": ["--directory", "C:/path/to/repo", "run", "job_matcher_mcp.py"]
    }
  }
}
```

### Via MCP Inspector
```bash
npx -y @modelcontextprotocol/inspector uv run job_matcher_mcp.py
```

## Tests
```bash
uv run pytest tests/ -v
```
