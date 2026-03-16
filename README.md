# Job Matcher MCP Server

STDIO based MCP server to collect and rank job postings data to be used later with skills and agents and save tokens.

## What it does
Builds a local MCP server (`job_matcher_mcp.py`) that orchestrates a smart pipeline:
1. **Scrape** — fetches jobs by keyword + location.
   - **Dynamic Domains**: Optimizes extraction for `gtm`, `sales`, `biotech`, or `data`.
   - **Multi-Source**: Automatically routes US queries to BuiltIn and European queries to LinkedIn.
2. **Embed** — computes and caches local HuggingFace embeddings for the descriptions.
3. **Filter & Score** — applies pre-filters (salary, seniority, relevancy) before ranking by cosine similarity.

## Requirements
- Python >= 3.11
- `uv` package manager
- `OPENAI_API_KEY` and `APIFY_API_TOKEN` (see Configuration)

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
```

Available `.env` settings:
- `DEFAULT_MIN_SALARY`: Filter out jobs below this base.
- `DEFAULT_EXCLUDE_SENIORITY`: Comma-separated list (e.g., `Intern,Junior`).
- `DEFAULT_EMPLOYMENT_TYPE`: (e.g., `Full-time,Contract`).
- `MAX_TOTAL_QUERIES`: Safety cap for parallel scraping (default: 8).

## Tools

- **`scrape_jobs`**: Parallel scraping with domain auto-detection.
- **`score_jobs`**: Computes local embeddings for a CSV.
- **`get_top_jobs`**: Ranks jobs against your `test_profile.json` with active filtering.
- **`list_saved_csvs`**: Utility to see all files in `data/` and `output/`.

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
