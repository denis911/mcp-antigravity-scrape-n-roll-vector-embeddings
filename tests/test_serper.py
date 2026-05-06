import pytest
import os
from unittest.mock import patch
from scrapers.serper_scraper import build_serper_queries, search_jobs_serper

def test_build_serper_queries():
    keywords = ["Data Scientist"]
    locations = ["Berlin"]
    ats_domains = ["boards.greenhouse.io", "jobs.lever.co"]
    queries = build_serper_queries(keywords, locations, ats_domains)
    assert len(queries) == 1
    # Check that ATS domains are included
    assert "site:boards.greenhouse.io" in queries[0]
    assert "site:jobs.lever.co" in queries[0]
    assert '"Data Scientist"' in queries[0]
    assert '"Berlin"' in queries[0]

@pytest.mark.asyncio
async def test_search_jobs_serper_dedup():
    mock_responses = [
        [
            {"title": "Job 1", "link": "http://example.com/job1", "snippet": "Snippet 1"},
            {"title": "Job 1 Duplicate", "link": "http://example.com/job1", "snippet": "Snippet 2"},
        ],
        [
            {"title": "Job 2", "link": "http://example.com/job2", "snippet": "Snippet 3"},
        ]
    ]

    # Mock the sync function
    with patch("scrapers.serper_scraper._serper_search_sync", side_effect=mock_responses):
        results = await search_jobs_serper(["query1", "query2"])
        
        assert len(results) == 2 # Deduped by link
        urls = {r["url"] for r in results}
        assert "http://example.com/job1" in urls
        assert "http://example.com/job2" in urls
        
        # Verify structure
        job1 = next(r for r in results if r["url"] == "http://example.com/job1")
        assert job1["source"] == "serper"
        assert job1["description_snippet"] == "Snippet 1"
