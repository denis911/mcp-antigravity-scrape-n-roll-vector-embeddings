"""
scrapers/keywords.py — Expanded Keyword Taxonomy
================================================
Curated sets of keywords for different job categories to optimize search yield.
"""

# Category A — Pre-sales / Solutions Engineering (HIGH PRIORITY)
KEYWORDS_PRESALES = [
    "pre-sales engineer",
    "solutions engineer",
    "solutions consultant",
    "sales engineer",
    "forward deployed engineer",
    "applied AI engineer",
    "technical sales engineer",
]

# Category B — Technology-based (not job-title-based)
KEYWORDS_TECH_BASED = [
    "AI automation sales",
    "LLM solutions engineer",
    "agentic AI sales",
    "GCP partner sales",
    "PostgreSQL solutions",
    "data infrastructure sales",
    "AI consulting presales",
]

# Category C — Customer Success / Technical AE / Renewals (MEDIUM PRIORITY)
KEYWORDS_CS_TAM = [
    "technical account manager AI",
    "customer success manager AI SaaS",
    "AI platform customer success",
    "enterprise customer success data",
    "technical customer success",
    "renewals account manager SaaS",
]

# Category D — FDE / Applied Engineering (STRETCH but FEASIBLE)
KEYWORDS_FDE = [
    "forward deployed engineer AI",
    "applied AI engineer",
    "implementation engineer AI SaaS",
    "solutions architect AI",
    "AI implementation consultant",
]

# Mapping for easy access
TAXONOMY = {
    "presales": KEYWORDS_PRESALES,
    "tech_based": KEYWORDS_TECH_BASED,
    "cs_tam": KEYWORDS_CS_TAM,
    "fde": KEYWORDS_FDE,
}
