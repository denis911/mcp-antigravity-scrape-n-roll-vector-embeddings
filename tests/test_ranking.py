import pandas as pd
from job_matcher_mcp import normalise_columns

def test_normalise_columns():
    df = pd.DataFrame(columns=["salary_json_min", "workplace_type_enum", "description_text "])
    df = normalise_columns(df)
    assert "salary_min" in df.columns
    assert "workplace_type" in df.columns
    assert "description_text" in df.columns
    assert "salary_json_min" not in df.columns
