import pytest
import pandas as pd
import json
import numpy as np
from unittest.mock import AsyncMock, patch, MagicMock, mock_open
from job_matcher_mcp import compare_searches

@pytest.mark.asyncio
async def test_compare_searches_logic():
    # Mock data for strategy A
    df_a = pd.DataFrame([
        {"title": "Job A1", "company": "Co A", "url": "url1", "description_text": "desc 1"},
        {"title": "Job A2", "company": "Co A", "url": "url2", "description_text": "desc 2"},
    ])
    
    # Mock data for strategy B
    df_b = pd.DataFrame([
        {"title": "Job B1", "company": "Co B", "url": "url3", "description_text": "desc 3"},
    ])
    
    # Mock profile
    mock_profile = {
        "summary": {"elevator_pitch": "pitch"},
        "identity": {"full_name": "Denis"}
    }
    
    # Mock profile JSON string
    profile_json = json.dumps(mock_profile)
    
    # Mock model and embeddings
    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([[0.1] * 384]) # dummy embedding
    
    with patch("job_matcher_mcp._run_scrape", new_callable=AsyncMock) as mock_scrape, \
         patch("job_matcher_mcp.get_model", return_value=mock_model), \
         patch("job_matcher_mcp.embed_texts") as mock_embed, \
         patch("pathlib.Path.exists", return_value=True), \
         patch("builtins.open", mock_open(read_data=profile_json)):
        
        # Mock embeddings to match df sizes
        mock_embed.side_effect = [
            [np.array([0.1]*384)] * 2, # for df_a
            [np.array([0.1]*384)] * 1  # for df_b
        ]
        
        # We need to control cosine_similarity separately for each call if possible, 
        # but for a simple test we can just mock it to return something plausible.
        
        # Setup side effects for _run_scrape
        mock_scrape.side_effect = [df_a, df_b]
        
        # Second call to cosine_similarity for strategy B
        with patch("job_matcher_mcp.cosine_similarity") as mock_sim:
            mock_sim.side_effect = [
                np.array([[0.6, 0.5]]), 
                np.array([[0.7]])
            ]
            
            strategies = [
                {"name": "A", "keywords": ["k1"], "locations": ["l1"]},
                {"name": "B", "keywords": ["k2"], "locations": ["l2"]}
            ]
            
            result = await compare_searches(strategies, profile_path="dummy.json")
            
            assert result["status"] == "ok"
            assert len(result["strategies"]) == 2
            assert result["winner"] == "B" # 0.7 > 0.6 (median)
            assert "recommendation" in result
            assert len(result["merged_top_10"]) > 0
            
            # Check if _run_scrape was called for both strategies
            assert mock_scrape.call_count == 2
