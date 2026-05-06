from scrapers.linkedin_scraper import _normalise_item, _normalise_seniority

def test_seniority_normalisation():
    assert _normalise_seniority("Mid-Senior level") == "Senior"
    assert _normalise_seniority("Entry level") == "Junior"
    assert _normalise_seniority("Director") == "Director"
    assert _normalise_seniority("Internship") == "Intern"
    assert _normalise_seniority("Not Applicable") is None
    assert _normalise_seniority("") is None

def test_url_fallback():
    item = {"jobUrl": "https://linkedin.com/jobs/view/123"}
    result = _normalise_item(item)
    assert result["url"] == "https://linkedin.com/jobs/view/123"

def test_url_fallback_link():
    item = {"link": "https://linkedin.com/jobs/view/456"}
    result = _normalise_item(item)
    assert result["url"] == "https://linkedin.com/jobs/view/456"

def test_source_tag():
    item = {"title": "Channel Manager", "url": "https://linkedin.com/jobs/view/789"}
    result = _normalise_item(item)
    assert result["source"] == "linkedin"

def test_description_fallback_chain():
    item = {"descriptionText": "Job description here"}
    result = _normalise_item(item)
    assert result["description_text"] == "Job description here"
    assert result["description"] == "Job description here"
