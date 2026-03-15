"""
job_scorer.py — Semantic Job-Profile Matcher
=============================================
Embeds job descriptions and candidate profile using HuggingFace sentence-transformers,
computes cosine similarity, and outputs a ranked CSV + YAML of top matches.

Usage:
    python job_scorer.py --jobs structured_jobs.csv --profile denis_kuramshin_context_v0.2.json
    python job_scorer.py --jobs structured_jobs.csv --profile denis_kuramshin_context_v0.2.json --top 10
    python job_scorer.py --jobs structured_jobs.csv --profile denis_kuramshin_context_v0.2.json --top 5 --output-yaml

Requirements:
    pip install sentence-transformers pandas numpy pyyaml beautifulsoup4

Model used by default: all-MiniLM-L6-v2
  - 80MB, fast, strong semantic similarity, no API key needed
  - Swap to 'BAAI/bge-large-en-v1.5' for higher accuracy at ~1.3GB
"""

import argparse
import json
import re
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ── Config ────────────────────────────────────────────────────────────────────

DEFAULT_MODEL = "all-MiniLM-L6-v2"
TEXT_COL = "description_text"       # primary text column in CSV
FALLBACK_COL = "description"        # fallback if primary is empty
EMBED_COL = "embedding"             # output: embedding stored as JSON string
SCORE_COL = "similarity_score"      # output: cosine similarity 0.0–1.0


# ── Text cleaning ─────────────────────────────────────────────────────────────

def clean_html(text: str) -> str:
    """Strip HTML tags and normalize whitespace."""
    if not text:
        return ""
    soup = BeautifulSoup(text, "html.parser")
    cleaned = soup.get_text(separator=" ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def get_job_text(row: pd.Series) -> str:
    """Extract best available text from a job row."""
    text = str(row.get(TEXT_COL, "") or "")
    if len(text) < 100:
        text = str(row.get(FALLBACK_COL, "") or "")
    # Also prepend title + company for better semantic signal
    prefix = f"{row.get('title', '')} at {row.get('company', '')}. "
    return clean_html(prefix + text)


# ── Profile → master text ─────────────────────────────────────────────────────

def profile_to_text(profile: dict) -> str:
    """
    Convert candidate JSON profile to a rich flat text for embedding.
    Includes: summary, skills, experience achievements, certs, projects.
    Weighted: recent experience and skills get repeated for emphasis.
    """
    parts = []

    # Identity and headline
    identity = profile.get("identity", {})
    summary = profile.get("summary", {})
    parts.append(summary.get("headline", ""))
    parts.append(summary.get("elevator_pitch", ""))

    # Tone/narrative
    tone = profile.get("tone_and_style", {})
    parts.extend(tone.get("things_to_emphasize", []))

    # Technical skills (repeated for weight)
    skills = profile.get("technical_skills", {})
    for category, items in skills.items():
        if isinstance(items, list):
            parts.append(" ".join(items))
            parts.append(" ".join(items))  # repeat for emphasis

    # Domain expertise
    domain = profile.get("domain_expertise", {})
    for category, items in domain.items():
        if isinstance(items, list):
            parts.extend(items)

    # Bioinformatics focus
    bio = domain.get("bioinformatics_focus", [])
    parts.extend(bio)

    # Experience — achievements from recent roles (last 4)
    experience = profile.get("experience", [])
    for role in experience[:4]:
        parts.append(f"{role.get('role', '')} at {role.get('company', '')}")
        parts.append(role.get("summary", ""))
        for achievement in role.get("achievements", []):
            parts.append(achievement.get("description", ""))
            parts.append(achievement.get("impact", ""))
        parts.extend(role.get("tech_used", []))

    # Certifications
    for cert in profile.get("certifications", []):
        parts.append(cert.get("name", ""))

    # Projects
    for proj in profile.get("projects", []):
        parts.append(proj.get("name", ""))
        parts.append(proj.get("description", ""))
        parts.extend(proj.get("tech_used", []))

    # ATS keywords (repeated — these are high signal)
    ats = profile.get("ats_keywords_pool", [])
    parts.extend(ats)
    parts.extend(ats)  # repeat for emphasis

    # Target roles
    open_roles = profile.get("open_roles_targeted", {})
    parts.extend(open_roles.get("primary", []))
    parts.extend(open_roles.get("secondary", []))

    # Education
    for edu in profile.get("education", []):
        parts.append(f"{edu.get('degree', '')} {edu.get('field', '')} {edu.get('institution', '')}")

    full_text = " ".join(str(p) for p in parts if p)
    full_text = re.sub(r"\s+", " ", full_text).strip()
    return full_text


# ── Embedding + scoring ───────────────────────────────────────────────────────

def embed_texts(texts: list[str], model: SentenceTransformer, batch_size: int = 64) -> np.ndarray:
    """Embed a list of texts, showing progress."""
    print(f"  Embedding {len(texts)} texts in batches of {batch_size}...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,   # L2-normalize → cosine = dot product
    )
    return embeddings


def scores_to_yaml(df_top: pd.DataFrame, profile: dict) -> str:
    """
    Convert top-N scored jobs to compact YAML for LLM agent consumption.
    Strips embeddings, includes only fields useful for CV generation.
    """
    output = {
        "candidate": {
            "name": profile.get("identity", {}).get("full_name", ""),
            "profile_version": profile.get("_meta", {}).get("version", ""),
        },
        "top_matches": []
    }

    keep_cols = [
        "title", "company", "location", "salary", "seniority",
        "workplace_type_enum", "tech_stack", "min_years_exp",
        "url", SCORE_COL, "description_text"
    ]

    for _, row in df_top.iterrows():
        job = {}
        for col in keep_cols:
            val = row.get(col, "")
            if col == "description_text":
                # Truncate description to save tokens
                text = str(val or "")[:800]
                job["description_snippet"] = text
            elif col == SCORE_COL:
                job["similarity_score"] = round(float(val), 4)
            elif val and str(val) not in ("nan", ""):
                job[col] = str(val)
        output["top_matches"].append(job)

    return yaml.dump(output, allow_unicode=True, sort_keys=False, default_flow_style=False)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Semantic job-profile matcher")
    parser.add_argument("--jobs", required=True, help="Path to jobs CSV file")
    parser.add_argument("--profile", required=True, help="Path to candidate JSON profile")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HuggingFace model name")
    parser.add_argument("--top", type=int, default=10, help="Number of top matches to return")
    parser.add_argument("--output-yaml", action="store_true", help="Also write top-N YAML for LLM agent")
    parser.add_argument("--output-csv", default=None, help="Output CSV path (default: jobs_scored.csv)")
    parser.add_argument("--min-score", type=float, default=0.0, help="Minimum similarity score filter")
    args = parser.parse_args()

    jobs_path = Path(args.jobs)
    profile_path = Path(args.profile)

    # ── Load data ──
    print(f"\n📂 Loading jobs from: {jobs_path}")
    df = pd.read_csv(jobs_path)
    print(f"   {len(df)} jobs loaded, {len(df.columns)} columns")

    print(f"📂 Loading profile from: {profile_path}")
    with open(profile_path, encoding="utf-8") as f:
        profile = json.load(f)
    candidate_name = profile.get("identity", {}).get("full_name", "Candidate")
    print(f"   Profile: {candidate_name} v{profile.get('_meta', {}).get('version', '?')}")

    # ── Clean job texts ──
    print("\n🧹 Cleaning job descriptions...")
    df["_clean_text"] = df.apply(get_job_text, axis=1)
    empty = (df["_clean_text"].str.len() < 50).sum()
    if empty > 0:
        print(f"   ⚠️  {empty} jobs have very short descriptions — they may score poorly")

    # ── Build profile text ──
    print("\n📝 Building candidate profile text...")
    profile_text = profile_to_text(profile)
    print(f"   Profile text: {len(profile_text)} chars, ~{len(profile_text.split())} words")

    # ── Load model ──
    print(f"\n🤖 Loading embedding model: {args.model}")
    print("   (First run downloads ~80MB — cached afterwards)")
    model = SentenceTransformer(args.model)

    # ── Embed ──
    print("\n⚡ Embedding job descriptions...")
    job_texts = df["_clean_text"].tolist()
    job_embeddings = embed_texts(job_texts, model)

    print("\n⚡ Embedding candidate profile...")
    profile_embedding = model.encode(
        [profile_text],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    # ── Score ──
    print("\n📊 Computing cosine similarity scores...")
    scores = cosine_similarity(profile_embedding, job_embeddings)[0]
    df[SCORE_COL] = scores

    # Store embeddings as JSON strings (optional — can be large)
    df[EMBED_COL] = [json.dumps(emb.tolist()) for emb in job_embeddings]

    # ── Filter and sort ──
    df_scored = df[df[SCORE_COL] >= args.min_score].copy()
    df_scored = df_scored.sort_values(SCORE_COL, ascending=False)
    df_top = df_scored.head(args.top)

    # ── Print results ──
    print(f"\n{'='*60}")
    print(f"🏆 TOP {args.top} MATCHES for {candidate_name}")
    print(f"{'='*60}")

    display_cols = ["title", "company", "location", "salary", SCORE_COL]
    for i, (_, row) in enumerate(df_top.iterrows(), 1):
        score = row[SCORE_COL]
        bar = "█" * int(score * 20)
        print(f"\n#{i:2d} [{score:.3f}] {bar}")
        print(f"     {row.get('title','')} @ {row.get('company','')}")
        print(f"     {row.get('location','')} | {row.get('salary','N/A')} | {row.get('seniority','')}")
        print(f"     Tech: {str(row.get('tech_stack',''))[:70]}")
        print(f"     URL: {row.get('url','')}")

    # ── Score distribution ──
    print(f"\n📈 Score distribution across all {len(df_scored)} jobs:")
    print(f"   Max:    {scores.max():.3f}")
    print(f"   Top 10: {sorted(scores)[-10]:.3f}+")
    print(f"   Median: {np.median(scores):.3f}")
    print(f"   Min:    {scores.min():.3f}")

    # ── Save CSV ──
    out_csv = args.output_csv or jobs_path.stem + "_scored.csv"
    # Drop embedding col from CSV if it bloats too much — keep scores only
    df_save = df_scored.drop(columns=["_clean_text"])
    df_save.to_csv(out_csv, index=False)
    print(f"\n💾 Scored CSV saved: {out_csv}")
    print(f"   Columns added: '{SCORE_COL}', '{EMBED_COL}'")

    # ── Save YAML for LLM agent ──
    if args.output_yaml:
        yaml_out = jobs_path.stem + f"_top{args.top}_for_llm.yaml"
        yaml_content = scores_to_yaml(df_top, profile)
        with open(yaml_out, "w", encoding="utf-8") as f:
            f.write(yaml_content)
        print(f"📄 LLM-ready YAML saved: {yaml_out}")
        # Estimate token savings
        full_tokens = sum(len(t.split()) for t in job_texts) // 0.75
        yaml_tokens = len(yaml_content.split()) // 0.75
        print(f"   Token estimate: {int(yaml_tokens):,} vs full CSV {int(full_tokens):,} "
              f"({100*yaml_tokens/full_tokens:.1f}% of original)")

    print(f"\n✅ Done.\n")


if __name__ == "__main__":
    main()
