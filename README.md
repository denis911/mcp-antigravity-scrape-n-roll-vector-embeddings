# Job Matcher MCP Server

STDIO-based MCP server to collect, extract, and semantically rank job postings data, intended to be used with LLM agents and skills while saving tokens.

## What it does
Builds a local MCP server (`job_matcher_mcp.py`) that orchestrates a smart pipeline:
1. **Scrape & Extract** — fetches jobs by keyword + location and uses an LLM to extract structured data.
   - **Dynamic Domains**: Auto-detects and optimizes extraction prompts for `gtm`, `sales`, `biotech`, or `data` roles.
   - **Multi-Backend**: Supports scraping via Apify (routing US queries to BuiltIn and EU to LinkedIn) or the Google Serper API targeting specific ATS platforms (Greenhouse, Lever, etc.) directly.
   - **Concurrent Processing**: Asynchronous HTML fetching and LLM extraction to drastically speed up parallel scrapes.
2. **Embed** — computes and caches local HuggingFace embeddings for the descriptions (runs locally, completely free).
3. **Filter & Score** — applies active pre-filters (salary floors, seniority exclusion, relevancy) before ranking by cosine similarity against your profile.

## Requirements
- Python >= 3.11
- `uv` package manager
- `OPENAI_API_KEY` (required for structured extraction and fit explanation)
- `APIFY_API_TOKEN` (if using Apify backend)
- `SERPER_API_KEY` (if using Serper backend)

## Installation (uv)

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
- `SCRAPER_BACKEND`: `apify` (default) or `serper` (uses Google Search footprints for ATS).
- `DEFAULT_JOB_DOMAIN`: `any`, `gtm`, `sales`, `biotech`, `data`.
- `DEFAULT_MIN_SALARY`: Filter out jobs below this base salary (e.g. `80000`).
- `DEFAULT_EXCLUDE_SENIORITY`: Comma-separated list (e.g. `Intern,Junior`).
- `DEFAULT_EMPLOYMENT_TYPE`: (e.g. `Full-time,Contract`).
- `MAX_TOTAL_QUERIES`: Safety cap for parallel scraping (default: 8).

## Tools

- **`scrape_jobs`**: Parallel scraping with domain auto-detection and multi-source routing.
- **`score_jobs`**: Computes local HuggingFace embeddings for a CSV (clean output without protocol corruption).
- **`get_top_jobs`**: Ranks jobs against your `test_profile.json` applying configurable pre-filters (salary, seniority, etc.).
- **`list_saved_csvs`**: Utility to view all cached CSV files in `data/` and `output/`. Backward compatible with older exports.

## Usage

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
