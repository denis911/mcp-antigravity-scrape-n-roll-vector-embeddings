# EXPAND-ABTest.md — Search Strategy A/B Testing + Keyword Expansion

## Goal

Two additions to the MCP server:

1. **`compare_searches()` tool** — runs multiple keyword sets, scores each batch
   independently, returns comparative stats (median score, top 5, score distribution)
   so you can objectively evaluate which search strategy finds better-fitting roles

2. **Expanded keyword taxonomy** — new job categories to scrape that Denis's
   profile matches but we haven't searched yet

---

## Part 1 — `compare_searches()` MCP Tool

### Why

We discovered duvo.ai Pre-Sales Engineer (A-grade, 4.80/5.0) by accident from BuiltIn.
It never appeared in our "account executive" / "channel manager" / "founding AE" scrapes
because we never searched for "pre-sales engineer" or "solutions engineer".

The problem: keyword-based scraping finds what you ask for and nothing else.
The solution: run multiple keyword strategies simultaneously and compare their yield.

### Tool signature

```python
@mcp.tool()
async def compare_searches(
    search_strategies: list[dict],   # list of {name, keywords, locations, job_domain}
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
            "locations": ["EMEA", "Remote", "London"],
            "job_domain": "sales"
        },
        {
            "name": "presales_se",
            "keywords": ["pre-sales engineer", "solutions engineer", "solutions consultant"],
            "locations": ["EMEA", "Remote", "London"],
            "job_domain": "sales"
        },
        {
            "name": "technology_based",
            "keywords": ["AI automation sales", "LLM solutions engineer"],
            "locations": ["EMEA", "Remote", "Europe"],
            "job_domain": "any"
        }
    ]

    Returns:
    {
        "strategies": [
            {
                "name": "presales_se",
                "jobs_scraped": 28,
                "jobs_scored": 22,
                "median_score": 0.521,
                "mean_score": 0.498,
                "max_score": 0.671,
                "score_distribution": {">=0.55": 6, "0.45-0.55": 8, "<0.45": 8},
                "top_5": [ {title, company, score, url}, ... ]
            },
            ...
        ],
        "winner": "presales_se",          # highest median score
        "recommendation": "presales_se returns 27% higher median score than traditional_sales",
        "merged_top_10": [ ... ],          # top 10 across ALL strategies combined, deduped
        "output_csvs": {"traditional_sales": "data/...", "presales_se": "data/..."}
    }
    """
```

### Implementation

```python
@mcp.tool()
async def compare_searches(
    search_strategies: list[dict],
    profile_path: str | None = None,
    max_results_per_query: int = 10,
    top_n_per_strategy: int = 5,
    output_dir: str | None = None,
) -> dict:
    profile_path = profile_path or os.getenv("DEFAULT_PROFILE_JSON")
    output_dir = output_dir or os.getenv("DEFAULT_OUTPUT_DIR", "output/")
    profile = json.loads(Path(profile_path).read_text(encoding="utf-8"))
    profile_text = profile_to_text(profile)
    profile_vec = _model.encode([profile_text], normalize_embeddings=True)[0]

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
        df = await _run_scrape(keywords, locations, max_results_per_query, job_domain, csv_path)

        if df.empty:
            results.append({"name": name, "jobs_scraped": 0, "error": "no results"})
            continue

        # Embed
        texts = df.apply(get_job_text, axis=1).tolist()
        vecs = embed_texts(texts, _model)
        df["embedding"] = [json.dumps(v.tolist()) for v in vecs]
        df["embedding_model"] = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")

        # Score
        scores = cosine_similarity([profile_vec], vecs)[0]
        df["similarity_score"] = scores
        df = df.sort_values("similarity_score", ascending=False).reset_index(drop=True)
        df.to_csv(csv_path, index=False)

        # Stats
        valid_scores = scores[~np.isnan(scores)]
        dist = {
            ">=0.55": int((valid_scores >= 0.55).sum()),
            "0.45-0.55": int(((valid_scores >= 0.45) & (valid_scores < 0.55)).sum()),
            "<0.45": int((valid_scores < 0.45).sum()),
        }

        top_5 = []
        for _, row in df.head(top_n_per_strategy).iterrows():
            top_5.append({
                "title": str(row.get("title", "")),
                "company": str(row.get("company", "")),
                "score": round(float(row["similarity_score"]), 4),
                "url": str(row.get("url", "")),
                "strategy": name,
            })
            all_jobs.append(top_5[-1])

        results.append({
            "name": name,
            "jobs_scraped": len(df),
            "median_score": round(float(np.median(valid_scores)), 4),
            "mean_score": round(float(np.mean(valid_scores)), 4),
            "max_score": round(float(valid_scores.max()), 4),
            "score_distribution": dist,
            "top_5": top_5,
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
        pct = round((best["median_score"] - second["median_score"]) / second["median_score"] * 100)
        recommendation = (
            f"'{best['name']}' returns {pct}% higher median score ({best['median_score']:.3f}) "
            f"than '{second['name']}' ({second['median_score']:.3f}). "
            f"Prioritise '{best['name']}' for future scrapes."
        )
    else:
        recommendation = "Only one strategy returned results — no comparison available."

    return {
        "strategies": results,
        "winner": winner,
        "recommendation": recommendation,
        "merged_top_10": merged_top10,
    }
```

---

## Part 2 — Keyword Taxonomy Expansion

New job categories to add to scraping rotation, with rationale for each.

### Category A — Pre-sales / Solutions Engineering (HIGH PRIORITY)

These are roles Denis is highly qualified for but we've never searched.
duvo.ai (4.80/5.0) is proof this category yields A-grade roles.

```python
KEYWORDS_PRESALES = [
    "pre-sales engineer",
    "solutions engineer",
    "solutions consultant",
    "sales engineer",
    "forward deployed engineer",
    "applied AI engineer",
    "technical sales engineer",
]
```

### Category B — Technology-based (not job-title-based)

Search by stack/domain rather than job function. Catches roles that don't use
standard titles but match Denis's profile.

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

### Category C — Customer Success / Technical AE / Renewals (MEDIUM PRIORITY)

Post-sales and hybrid commercial/technical roles. Lower quota pressure,
higher relationship and technical depth. Feasible for Denis given delivery
experience at EDB and Dbvisit.

```python
KEYWORDS_CS_TAM = [
    "technical account manager AI",
    "customer success manager AI SaaS",
    "AI platform customer success",
    "enterprise customer success data",
    "technical customer success",
    "renewals account manager SaaS",
]
```

**Feasibility assessment for Denis:**
- Customer Success Manager: ✅ Feasible — post-deployment consulting, adoption tracking,
  expansion identification are things Denis has done throughout his career at EDB and Dbvisit.
  Not a stretch; just different title.
- Technical Account Manager: ✅ Strong fit — technical depth + commercial instinct +
  C-level relationship building. Many AI companies pay TAMs competitively.
- Renewals AE: ⚠️ Feasible but lower ceiling — retention-focused, less new business hunting.
  Good for stability, less interesting commercially.

### Category D — FDE / Applied Engineering (STRETCH but FEASIBLE)

Forward Deployed Engineer roles at AI companies. Requires coding competency
but not senior SWE depth. Denis's profile: Python scripting + GCP deployment +
agentic systems in production = qualifies.

```python
KEYWORDS_FDE = [
    "forward deployed engineer AI",
    "applied AI engineer",
    "implementation engineer AI SaaS",
    "solutions architect AI",
    "AI implementation consultant",
]
```

**Feasibility for Denis:** duvo.ai explicitly listed FDE as a likely background for
their PSE role. At 32-person AI-native startups, the line between PSE and FDE is thin.
Denis's agentic pipeline project + GCP certs + prototyping track record = minimum
viable FDE profile. Test with small scrape before committing time to applications.

---

## Part 3 — Recommended A/B Test to Run First

Compare three strategies simultaneously. Answers the question: which keyword
approach yields the highest-scoring roles for Denis?

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

Expected outcome hypothesis:
- `presales_se` will have higher median score than `traditional_sales`
  because the PSE/SE motion maps more precisely to Denis's technical+commercial
  dual track than pure AE roles
- `tech_based` will have higher MAX score but higher variance
  (catches more niche exact-match roles but also more noise)

---

## Part 4 — SKILL.md Updates Needed

After implementing `compare_searches()`, update SKILL.md with:

1. **New tool entry in MCP tools table:**
   ```
   | compare_searches() | Run A/B test across keyword strategies, return winner + merged top 10 |
   ```

2. **New section: "Keyword Strategy"** — document the four categories above
   so a fresh conversation knows what to search for beyond the defaults

3. **Update DEFAULT_KEYWORDS** in `.env`:
   ```
   DEFAULT_KEYWORDS_PRESALES=pre-sales engineer,solutions engineer,solutions consultant
   DEFAULT_KEYWORDS_CS=technical account manager AI,customer success manager AI SaaS
   DEFAULT_KEYWORDS_FDE=forward deployed engineer AI,applied AI engineer
   ```

---

## Build Order for Coding Agent

1. Add `compare_searches()` to `job_matcher_mcp.py` — uses existing
   `scrape_jobs()`, `score_jobs()`, `get_top_jobs()` internals
2. Extract `_run_scrape()` as a shared internal function called by both
   `scrape_jobs()` and `compare_searches()`
3. Add keyword taxonomy constants to `.env` or a new `keywords.py` config file
4. Update SKILL.md with new tool and keyword categories
5. Run the recommended A/B test above as first live validation
