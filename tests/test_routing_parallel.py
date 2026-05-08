import pytest
import pandas as pd
import asyncio
from unittest.mock import AsyncMock, patch
from job_matcher_mcp import _run_scrape

@pytest.mark.asyncio
async def test_run_scrape_parallel_defaults():
    """Verify that all three main scrapers are called by default."""
    keywords = ["kw1"]
    locations = ["loc1"]
    
    with patch("job_matcher_mcp.scrape_builtin", new_callable=AsyncMock) as m_builtin, \
         patch("job_matcher_mcp.scrape_linkedin", new_callable=AsyncMock) as m_linkedin, \
         patch("job_matcher_mcp.scrape_serper", new_callable=AsyncMock) as m_serper:
        
        m_builtin.return_value = pd.DataFrame([{"url": "u1", "title": "b1"}])
        m_linkedin.return_value = pd.DataFrame([{"url": "u2", "title": "l1"}])
        m_serper.return_value = pd.DataFrame([{"url": "u3", "title": "s1"}])
        
        df = await _run_scrape(keywords, locations)
        
        assert m_builtin.called
        assert m_linkedin.called
        assert m_serper.called
        assert len(df) == 3

@pytest.mark.asyncio
async def test_run_scrape_explicit_scrapers():
    """Verify that only requested scrapers are called."""
    keywords = ["kw1"]
    locations = ["loc1"]
    
    with patch("job_matcher_mcp.scrape_builtin", new_callable=AsyncMock) as m_builtin, \
         patch("job_matcher_mcp.scrape_linkedin", new_callable=AsyncMock) as m_linkedin:
        
        m_builtin.return_value = pd.DataFrame([{"url": "u1"}])
        
        df = await _run_scrape(keywords, locations, scrapers=["builtin"])
        
        assert m_builtin.called
        assert not m_linkedin.called
        assert len(df) == 1

@pytest.mark.asyncio
async def test_run_scrape_japan_autodetect():
    """Verify that Japan scraper is added automatically for Japan locations."""
    keywords = ["kw1"]
    locations = ["Tokyo"] # in DEFAULT_JAPAN_LOCATIONS
    
    with patch("job_matcher_mcp.scrape_builtin", new_callable=AsyncMock) as m_builtin, \
         patch("job_matcher_mcp.scrape_japan", new_callable=AsyncMock) as m_japan:
        
        m_builtin.return_value = pd.DataFrame()
        m_japan.return_value = pd.DataFrame([{"url": "uj1"}])
        
        df = await _run_scrape(keywords, locations)
        
        assert m_japan.called
        assert len(df) == 1

@pytest.mark.asyncio
async def test_run_scrape_deduplication():
    """Verify that jobs with same URL are deduplicated across scrapers."""
    keywords = ["kw1"]
    locations = ["loc1"]
    
    with patch("job_matcher_mcp.scrape_builtin", new_callable=AsyncMock) as m_builtin, \
         patch("job_matcher_mcp.scrape_linkedin", new_callable=AsyncMock) as m_linkedin:
        
        m_builtin.return_value = pd.DataFrame([{"url": "shared_url", "title": "b1"}])
        m_linkedin.return_value = pd.DataFrame([{"url": "shared_url", "title": "l1"}])
        
        df = await _run_scrape(keywords, locations, scrapers=["builtin", "linkedin"])
        
        assert len(df) == 1
        assert df.iloc[0]["title"] == "b1" # BuiltIn comes first in our logic
