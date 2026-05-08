# EXPAND-ABTest.md — Search Strategy A/B Testing + Keyword Expansion

## Goal

Three additions to the MCP server:

1. **`compare_searches()` tool** — A/B test multiple keyword strategies,
   compare median/max scores, return winner + merged top 10

2. **`snipe_url()` tool** — fetch, parse, embed, and score any single job URL
   instantly without running a full scrape pipeline (~$0.002, ~5 seconds)

3. **Expanded keyword taxonomy** — new job categories Denis hasn't searched yet

---

## Part 1 — `compare_searches()` MCP Tool

Already implemented in `job_matcher_mcp.py`. See existing code.

### Usage example

```python
compare_searches(
    search_strategies=[
        {
            "name": "traditional_sales",
            "keywords": ["account executive", "channel manager", "founding AE"],
            "locations": ["EMEA", "Remote", "Europe"],
            "job_domain": "sales"
        },
        {
            "name": "presales_se",
            "keywords": ["pre-sales engineer", "solutions engineer", "solutions consultant"],
            "locations": ["EMEA", "Remote", "Europe"],
            "job_domain": "sales"
        },
        {
            "name": "tech_based",
            "keywords": ["AI automation presales", "LLM solutions engineer", "agentic AI sales"],
            "locations": ["EMEA", "Remote", "Europe"],
            "job_domain": "any"
        }
    ],
    max_results_per_query=10,
    top_n_per_strategy=5
)
```

### Returns

```json
{
    "strategies": [
        {
            "name": "presales_se",
            "jobs_scraped": 28,
            "median_score": 0.521,
            "mean_score": 0.498,
            "max_score": 0.671,
            "score_distribution": {">=0.55": 6, "0.45-0.55": 8, "<0.45": 8},
            "top_jobs": [...]
        }
    ],
    "winner": "presales_se",
    "recommendation": "presales_se returns 27% higher median score...",
    "merged_top_10": [...]
}
```

---

## Part 2 — `snipe_url()` MCP Tool

### Why

When a specific job URL is found manually (BuiltIn, LinkedIn, company careers page,
colleague recommendation), there is no way to feed it directly into the pipeline
without copy-pasting the JD. `snipe_url()` accepts any URL, fetches the full JD,
runs LLM extraction, embeds, scores against the profile, and returns a structured
result — all in one call.

**Origin story:** duvo.ai Pre-Sales Engineer (A-grade, 4.80/5.0) was found manually
on BuiltIn. It never appeared in any keyword scrape. `snipe_url()` prevents this
from happening again — any URL can be scored in seconds.

**Cost:** ~$0.002 per URL. Essentially free.

### Tool signature

```python
@mcp.tool()
async def snipe_url(
    url: str,
    profile_path: str | None = None,
    explain: bool = True,
) -> dict:
    """
    Fetch, parse, embed, and score a single job posting URL.
    Bypasses scraping pipeline — direct URL to scored result in one call.

    Supports:
    - Direct HTTP: BuiltIn, Lever, Greenhouse, Welcome to the Jungle, careers pages
    - Apify fallback: LinkedIn, Glassdoor, and other bot-protected sites

    Parameters:
    - url: Any job posting URL
    - profile_path: Path to candidate JSON profile (default from .env)
    - explain: If True, generates fit explanation via gpt-4o-mini

    Returns:
    {
        "status": "ok",
        "title": "Pre-Sales Engineer",
        "company": "duvo.ai",
        "location": "Prague, CZE",
        "salary": "...",
        "seniority": "Senior",
        "tech_stack": [...],
        "url": "https://...",
        "similarity_score": 0.671,
        "fit_explanation": "Strong match: Denis's agentic AI...",
        "description_text": "...",
        "source": "snipe"
    }
    """
```

### Implementation — add to `job_matcher_mcp.py`

```python
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
```

### Notes for coding agent

- **Direct HTTP first** — BuiltIn, Lever, Greenhouse, Welcome to the Jungle all allow
  direct fetches. Try this first — it is free and instant.
- **Apify actor** — use `apify/cheerio-scraper` (lighter and cheaper than
  `apify/web-scraper`). Cost ~$0.001 per URL.
- **LinkedIn** — always requires Apify fallback (robots.txt blocks direct fetch).
- **No CSV written** — returns JSON only. Ephemeral by design.
- **`_explain_job()` signature** — matches the existing implementation in
  `job_matcher_mcp.py` which takes `(client, model, candidate_summary, row)`.
- **`ensure_canonical_columns`** — import from `extractor.py` same as other tools.

### Validation test

After implementation, restart Claude Desktop and run:

```python
snipe_url("https://builtin.com/job/pre-sales-engineer-europe-based/9256587")
```

Expected result:
- `title`: contains "Pre-Sales Engineer"
- `company`: "duvo.ai" (or similar)
- `similarity_score`: ~0.65–0.70
- `fit_explanation`: mentions agentic AI, Prague, presales motion

---

## Part 3 — Keyword Taxonomy Expansion

### Category A — Pre-sales / Solutions Engineering (HIGH PRIORITY)

duvo.ai (4.80/5.0) and Tiro/Agentic AI PSE (0.608) both came from this category.
We had never searched for these terms before.

```python
KEYWORDS_PRESALES = [
    "pre-sales engineer",
    "solutions engineer",
    "solutions consultant",
    "sales engineer",
    "forward deployed engineer",
    "applied AI engineer",
    "technical sales engineer",
    "AI solutions consultant",
]
```

### Category B — Technology-based search

Search by stack/domain rather than job function.

```python
KEYWORDS_TECH_BASED = [
    "AI automation sales",
    "LLM solutions engineer",
    "agentic AI sales",
    "GCP partner sales",
    "PostgreSQL solutions",
    "data infrastructure sales",
    "AI consulting presales",
]
```

### Category C — Customer Success / TAM (MEDIUM PRIORITY)

```python
KEYWORDS_CS_TAM = [
    "technical account manager AI",
    "customer success manager AI SaaS",
    "AI platform customer success",
    "enterprise customer success data",
    "technical customer success",
]
```

### Category D — FDE / Applied Engineering (STRETCH)

```python
KEYWORDS_FDE = [
    "forward deployed engineer AI",
    "applied AI engineer",
    "implementation engineer AI SaaS",
    "solutions architect AI",
    "AI implementation consultant",
]
```

---

## Part 4 — SKILL.md Updates Needed

Add to MCP tools table in SKILL.md:

```
| compare_searches() | A/B test keyword strategies, return winner + merged top 10 |
| snipe_url(url)     | Score any single job URL instantly. ~$0.002. No CSV needed. |
```

Add keyword strategy section with four categories above.

Update `.env`:
```
DEFAULT_KEYWORDS_PRESALES=pre-sales engineer,solutions engineer,solutions consultant
DEFAULT_KEYWORDS_CS=technical account manager AI,customer success manager AI SaaS
DEFAULT_KEYWORDS_FDE=forward deployed engineer AI,applied AI engineer
```

---

## Build Order for Coding Agent

1. `snipe_url()` — add to `job_matcher_mcp.py` as `@mcp.tool()`.
   Full implementation above. Uses existing `get_model()`, `get_job_text()`,
   `profile_to_text()`, `_explain_job()`, `extract_structured_data_async()`.
   Test with BuiltIn URL before LinkedIn URL.

2. `compare_searches()` — already implemented. Verify it works with A/B test run.

3. Keyword taxonomy — add to `.env` as `DEFAULT_KEYWORDS_*` constants.

4. Update README.md with new tools and keyword categories.

5. Validation:
   - `snipe_url("https://builtin.com/job/pre-sales-engineer-europe-based/9256587")`
     → should return duvo.ai with score ~0.67
   - `compare_searches([traditional_sales, presales_se, tech_based])`
     → should confirm presales_se has higher median score
