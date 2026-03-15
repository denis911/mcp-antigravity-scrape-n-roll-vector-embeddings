# Job Matcher MCP Server

STDIO based MCP server to collect and rank job postings data to be used later with skills and agents and save tokens.

## What it does
Builds a local MCP server (`job_matcher_mcp.py`) that orchestrates a three-stage pipeline:
1. **Scrape** — fetches live job postings by keyword + location and saves raw CSV
2. **Embed** — computes and caches HuggingFace embeddings per job description
3. **Score & Rank** — scores cosine similarity vs. candidate profile, provides LLM fit explanations, returning formatted JSON

## Requirements
- Python >= 3.11
- `uv` package manager

## Installation (uv)

This project uses `uv` to manage dependencies seamlessly without polluting system python.

Run setup and install dependencies:
```bash
uv sync
```

## Configuration (.env)

API Keys are expected in your Windows User variables. You can verify they are ready to use via PowerShell:

```powershell
echo $env:OPENAI_API_KEY
echo $env:APIFY_API_TOKEN
```

If not, please add `OPENAI_API_KEY` and `APIFY_API_TOKEN` to your user environment variables and restart your terminal.

> [!NOTE] 
> No **HuggingFace API token** is required to run this server. The embeddings are computed using the public `all-MiniLM-L6-v2` model from the `sentence-transformers` library, which automatically downloads and caches the model locally (~80MB) on the first run.

## Usage

### Via Claude Desktop
Add to your `claude_desktop_config.json`:
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

### Via MCP Inspector (manual testing)
To manually test and interact with the tools during development, rely on the official Node-based inspector to connect strictly to the `uv` run environment:
```bash
npx -y @modelcontextprotocol/inspector uv run job_matcher_mcp.py
```

1. Run **`scrape_jobs`** with a max limit: `{"max_results_per_query": 10}`. Note the `csv_path` it returns.
2. Run **`score_jobs`** providing the CSV path from step 1: `{"csv_path": "data/raw_jobs_...csv"}`. This step computes the embeddings and saves them in-place.
3. Run **`get_top_jobs`** providing the embedded CSV: `{"csv_path": "data/raw_jobs_...csv"}`. Wait for the OpenAI summaries to generate.

## Tests
To run the automated test suite for scoring mechanisms and pipeline correctness:
```bash
uv run pytest tests/ -v
```
