from japan_scraper import build_japan_queries, DEFAULT_JAPAN_BOARDS

def test_no_location_in_queries():
    """Japan queries must NOT contain location strings."""
    queries = build_japan_queries(
        keywords=["EMEA sales", "Europe channel partner"],
        boards=DEFAULT_JAPAN_BOARDS,
    )
    for q in queries:
        assert "Tokyo" not in q
        assert "Japan" not in q
        assert "Berlin" not in q

def test_query_contains_keyword():
    queries = build_japan_queries(["EMEA sales"], DEFAULT_JAPAN_BOARDS)
    assert any("EMEA sales" in q for q in queries)

def test_query_contains_all_boards():
    queries = build_japan_queries(["EMEA sales"], DEFAULT_JAPAN_BOARDS)
    for board in DEFAULT_JAPAN_BOARDS:
        assert any(board in q for q in queries)

def test_one_query_per_keyword():
    queries = build_japan_queries(["kw1", "kw2", "kw3"], DEFAULT_JAPAN_BOARDS)
    assert len(queries) == 3   # one per keyword, not keyword × location
