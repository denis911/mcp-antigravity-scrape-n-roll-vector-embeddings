import json
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from job_scorer import get_job_text, profile_to_text, embed_texts

def load_test_profile():
    with open("tests/fixtures/test_profile.json", "r", encoding="utf-8") as f:
        return json.load(f)

model = SentenceTransformer("all-MiniLM-L6-v2")
df = pd.read_csv("tests/fixtures/sample_jobs.csv")
df["_clean_text"] = df.apply(get_job_text, axis=1)

text = profile_to_text(load_test_profile())
profile_embedding = model.encode([text], normalize_embeddings=True, convert_to_numpy=True)

job_embeddings = embed_texts(df["_clean_text"].tolist(), model)
df["similarity_score"] = cosine_similarity(profile_embedding, job_embeddings)[0]

print("--- TOP JOBS ---")
print(df[df["expected_rank_tier"] == "top"][["title", "similarity_score"]])
print("--- MID JOBS ---")
print(df[df["expected_rank_tier"] == "mid"][["title", "similarity_score"]])
print("--- BOTTOM JOBS ---")
print(df[df["expected_rank_tier"] == "bottom"][["title", "similarity_score"]])
