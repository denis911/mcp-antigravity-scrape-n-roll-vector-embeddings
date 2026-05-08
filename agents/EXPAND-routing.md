# EXPAND-routing.md — Simplified Multi-Scraper Routing

## Problem with Current Routing

The current routing logic in `_run_scrape()` assigns each location to exactly one scraper:

```
US cities → BuiltIn only
EU cities → LinkedIn only  
Japan → Japan boards only
Everything else → Serper only
```

This is wrong for several reasons proven in practice:

1. **duvo.ai (A-grade, 4.80/5.0)** was found on BuiltIn for a Prague/EU role.
   It never appeared on LinkedIn or Serper. BuiltIn-exclusive discovery.

2. **"Czech Republic" location** routes to Serper (fallback) instead of BuiltIn
   because the routing logic fails to match it. A hardcoded fix was proposed
   but rejected as poor design.

3. **LinkedIn misses some roles** — companies avoid LinkedIn job postings to
   reduce spam applications, but still post on BuiltIn or their own ATS.

4. **Serper catches ATS boards** (Greenhouse, Lever, Ashby, Wellfound) that
   neither BuiltIn nor LinkedIn index well.

5. **The assumption "BuiltIn is US-only" is false.** BuiltIn indexes global
   companies, remote roles, and EU-based startups. Its `?country=CZE` endpoint
   returns 2,232 Czech Republic jobs.

**Root cause:** Location-based routing is a heuristic that worked for simple
US vs EU separation but breaks for any non-standard case.

---

## Proposed Solution — Parallel Multi-Scraper Mode

Replace location-based routing with a **scraper selection** parameter.
Default: run all scrapers in parallel for maximum coverage.
Optional: restrict to specific scrapers for speed or cost control.

---

## API Change — `scrape_jobs()` new parameter

```python
@mcp.tool()
async def scrape_jobs(
    keywords: list[str] | None = None,
    locations: list[str] | None = None,
    max_results_per_query: int = 15,       # lower default for parallel runs
    job_domain: str | None = None,
    max_total_queries: int | None = None,
    output_path: str | None = None,
    ats_domains: list[str] | None = None,
    scrapers: list[str] | None = None,     # NEW — ["builtin", "linkedin", "serper", "japan"]
                                            # None = all scrapers (default)
) -> dict:
    """
    ...
    - scrapers: (Optional) List of scrapers to use. Default: all available.
      Options: "builtin", "linkedin", "serper", "japan"
      Examples:
        scrapers=["builtin"]           # BuiltIn only (fast, US + global)
        scrapers=["linkedin", "serper"] # EU-focused
        scrapers=["builtin", "linkedin", "serper"]  # all except Japan (default for non-Japan)
      If None, uses all scrapers (builtin + linkedin + serper + japan if Japan location detected)
    """
```

---

## New `_run_scrape()` Logic

Replace the current location-routing if/elif/else block with parallel scraper execution:

```python
async def _run_scrape(
    keywords: list[str],
    locations: list[str],
    max_results_per_query: int = 15,
    job_domain: str | None = None,
    max_total_queries: int | None = None,
    output_path: str | None = None,
    ats_domains: list[str] | None = None,
    scrapers: list[str] | None = None,     # NEW
) -> pd.DataFrame:

    # Determine which scrapers to run
    # Default: all three main scrapers
    # Japan added automatically if any location matches DEFAULT_JAPAN_LOCATIONS
    if scrapers is None:
        active_scrapers = ["builtin", "linkedin", "serper"]
    else:
        active_scrapers = [s.lower() for s in scrapers]

    # Auto-add Japan scraper if Japan location detected
    jp_locations = [l for l in locations if any(
        l.lower() in j.lower() for j in DEFAULT_JAPAN_LOCATIONS
    )]
    non_jp_locations = [l for l in locations if l not in jp_locations]

    # Safety cap — divide budget across active scrapers
    n_scrapers = len(active_scrapers) + (1 if jp_locations else 0)
    per_scraper_queries = max(1, (max_total_queries or int(os.getenv("MAX_TOTAL_QUERIES", 8))) // n_scrapers)

    dfs = []

    # Run all active scrapers in parallel on the same keywords × locations
    tasks = []

    if "builtin" in active_scrapers and non_jp_locations:
        tasks.append(
            scrape_builtin(keywords, non_jp_locations, max_results_per_query, job_domain=job_domain)
        )

    if "linkedin" in active_scrapers and non_jp_locations:
        tasks.append(
            scrape_linkedin(keywords, non_jp_locations, max_results_per_query, job_domain=job_domain)
        )

    if "serper" in active_scrapers and non_jp_locations:
        tasks.append(
            scrape_serper(keywords, non_jp_locations, max_results_per_query,
                         job_domain=job_domain, ats_domains=ats_domains)
        )

    if jp_locations:
        tasks.append(
            scrape_japan(keywords, jp_locations, max_results_per_query, job_domain=job_domain)
        )

    # Execute all scrapers concurrently
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for result in results:
        if isinstance(result, Exception):
            logging.error(f"Scraper error: {result}")
            continue
        if isinstance(result, pd.DataFrame) and not result.empty:
            dfs.append(result)

    if not dfs:
        return pd.DataFrame()

    # Merge and deduplicate by URL
    df = pd.concat(dfs, ignore_index=True)
    df = normalise_columns(df)
    if "url" in df.columns:
        df = df.drop_duplicates(subset=["url"], keep="first")
        logging.info(f"After dedup: {len(df)} unique jobs from {len(dfs)} scrapers")

    df.to_csv(out_path, index=False)
    return df
```

---

## Credit/Cost Management

Running all three scrapers in parallel multiplies Apify cost by ~3x vs single scraper.
Manage with `max_results_per_query` — lower default (15 instead of 100) for parallel runs.

Recommended defaults:

| Use case | scrapers | max_results_per_query | Expected jobs |
|---|---|---|---|
| Broad discovery | None (all) | 10-15 | 30-60 per keyword |
| Czech Republic deep search | ["builtin"] | 50 | 50+ from CZE catalogue |
| EU-focused, cost-sensitive | ["linkedin", "serper"] | 15 | 30+ |
| US remote only | ["builtin"] | 20 | 20 |
| Maximum coverage | None (all) | 20 | 60-80 per keyword |

```python
# Broad AI sales search — all scrapers, moderate volume
scrape_jobs(
    keywords=["account executive", "solutions engineer"],
    locations=["EMEA", "Remote", "Czech Republic"],
    job_domain="sales",
    max_results_per_query=15,
    max_total_queries=6
)

# Czech Republic deep dive — BuiltIn country catalogue only
scrape_jobs(
    keywords=["account executive", "channel manager", "solutions consultant"],
    locations=["Czech Republic"],
    scrapers=["builtin"],
    max_results_per_query=50,
    max_total_queries=3
)

# EU LinkedIn + Serper, skip BuiltIn
scrape_jobs(
    keywords=["founding account executive"],
    locations=["London", "Berlin", "Remote"],
    scrapers=["linkedin", "serper"],
    max_results_per_query=20,
    max_total_queries=6
)
```

---

## Deduplication — Already Handled

`scrape_apify()` in `builtin_scraper.py` already deduplicates by URL within a single
scraper. The new `_run_scrape()` deduplicates across scrapers using:

```python
df = df.drop_duplicates(subset=["url"], keep="first")
```

Same job posted on both BuiltIn and LinkedIn will appear once.
Source column preserved — you can see which scraper found each job:
```
source = "builtin" | "linkedin" | "serper" | "japan_boards" | "snipe"
```

---

## Remove Routing Logic Entirely

Delete the following from `_run_scrape()`:

```python
# DELETE ALL OF THIS:
s_map = source_map or DEFAULT_SOURCE_MAP
builtin_locs = s_map.get("builtin", [])
linkedin_locs = s_map.get("linkedin", [])

us_locations = [l for l in locations if any(l.lower() in x.lower() for x in builtin_locs)]
jp_locations = [l for l in locations if any(l.lower() in j.lower() for j in DEFAULT_JAPAN_LOCATIONS)]
li_locations = [l for l in locations if any(l.lower() in x.lower() for x in linkedin_locs) and l not in jp_locations]
eu_locations = [l for l in locations if l not in us_locations and l not in jp_locations and l not in li_locations]

if us_locations:
    df_us = await scrape_builtin(...)
if li_locations:
    df_li = await scrape_linkedin(...)
if eu_locations:
    df_eu = await scrape_serper(...)
if jp_locations:
    df_jp = await scrape_japan(...)
```

Replace with the parallel execution block above.

Also remove `source_map` parameter from `scrape_jobs()` and `_run_scrape()` —
no longer needed since routing is gone.

---

## `DEFAULT_SOURCE_MAP` in `builtin_scraper.py`

Keep `DEFAULT_SOURCE_MAP` for backward compatibility (other code may reference it)
but it no longer drives routing in `_run_scrape()`. Can be repurposed as documentation
of which locations each scraper was originally optimised for, or removed in a future cleanup.

The `COUNTRY_CODES` dict in `build_urls()` stays — it correctly generates the
`?country=CZE&allLocations=true` URL format for BuiltIn country-based searches.
This is a genuine improvement independent of routing changes.

---

## `compare_searches()` — No Change Needed

`compare_searches()` calls `_run_scrape()` internally. Once `_run_scrape()` supports
the `scrapers` parameter, `compare_searches()` can optionally pass it through:

```python
# Optional addition to compare_searches strategy dict:
{
    "name": "builtin_only",
    "keywords": ["account executive"],
    "locations": ["Czech Republic"],
    "job_domain": "sales",
    "scrapers": ["builtin"]   # optional — restrict to specific scrapers
}
```

---

## SKILL.md Update Needed

After implementation, update the MCP tools table:

```
| scrape_jobs() | ... scrapers=["builtin","linkedin","serper"] controls which 
|               | scrapers run. Default: all three. Japan added automatically
|               | for Japanese locations. |
```

Update the workflow section:
```
# Broad discovery — all scrapers, same keywords, deduplicated output
scrape_jobs(
    keywords=["account executive", "solutions engineer"],
    locations=["EMEA", "Remote", "Czech Republic"],
    job_domain="sales"
)

# Czech Republic deep dive
scrape_jobs(
    keywords=["account executive"],
    locations=["Czech Republic"],
    scrapers=["builtin"]
)
```

---

## Build Order for Coding Agent

1. Add `scrapers: list[str] | None = None` parameter to `_run_scrape()` and `scrape_jobs()`
2. Replace routing if/elif/else block in `_run_scrape()` with parallel `asyncio.gather()` block
3. Add URL deduplication after `pd.concat()` in `_run_scrape()`
4. Remove `source_map` parameter from `scrape_jobs()` (keep in `_run_scrape()` signature
   temporarily for backward compatibility, but stop using it for routing)
5. Update `compare_searches()` to optionally pass `scrapers` through from strategy dict
6. Update SKILL.md with new parameter and usage examples
7. Test with:
   - `scrape_jobs(keywords=["account executive"], locations=["Czech Republic"])` 
     → should return jobs from all 3 scrapers with `source` mix of builtin/linkedin/serper
   - `scrape_jobs(keywords=["account executive"], locations=["Czech Republic"], scrapers=["builtin"])`
     → should return BuiltIn CZE country catalogue results only
