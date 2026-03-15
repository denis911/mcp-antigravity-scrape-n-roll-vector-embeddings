import json
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from job_scorer import get_job_text, profile_to_text, embed_texts

def load_test_profile():
    with open("tests/fixtures/test_profile.json", "r", encoding="utf-8") as f:
        return json.load(f)

def test_profile_to_text_nonempty():
    text = profile_to_text(load_test_profile())
    assert len(text) > 200
    assert any(kw in text.lower() for kw in ["python", "machine learning", "llm", "gtm"])

def test_top_jobs_outscore_bottom():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    df = pd.read_csv("tests/fixtures/sample_jobs.csv")
    df["_clean_text"] = df.apply(get_job_text, axis=1)
    
    text = profile_to_text(load_test_profile())
    profile_embedding = model.encode([text], normalize_embeddings=True, convert_to_numpy=True)
    
    job_embeddings = embed_texts(df["_clean_text"].tolist(), model)
    scores = cosine_similarity(profile_embedding, job_embeddings)[0]
    df["similarity_score"] = scores
    
    top_mean = df[df["expected_rank_tier"] == "top"]["similarity_score"].mean()
    bottom_mean = df[df["expected_rank_tier"] == "bottom"]["similarity_score"].mean()
    
    assert top_mean > bottom_mean + 0.05

def test_top_jobs_in_top10():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    df = pd.read_csv("tests/fixtures/sample_jobs.csv")
    df["_clean_text"] = df.apply(get_job_text, axis=1)
    
    text = profile_to_text(load_test_profile())
    profile_embedding = model.encode([text], normalize_embeddings=True, convert_to_numpy=True)
    
    job_embeddings = embed_texts(df["_clean_text"].tolist(), model)
    df["similarity_score"] = cosine_similarity(profile_embedding, job_embeddings)[0]
    
    top_10 = df.sort_values("similarity_score", ascending=False).head(10)
    
    top_tier_in_top_10 = top_10[top_10["expected_rank_tier"] == "top"].shape[0]
    assert top_tier_in_top_10 >= 4

def test_score_range():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    df = pd.read_csv("tests/fixtures/sample_jobs.csv")
    df["_clean_text"] = df.apply(get_job_text, axis=1)
    
    text = profile_to_text(load_test_profile())
    profile_embedding = model.encode([text], normalize_embeddings=True, convert_to_numpy=True)
    
    job_embeddings = embed_texts(df["_clean_text"].tolist(), model)
    scores = cosine_similarity(profile_embedding, job_embeddings)[0]
    
    assert -0.0001 <= scores.min() and scores.max() <= 1.0001
    assert not np.isnan(scores).any()
