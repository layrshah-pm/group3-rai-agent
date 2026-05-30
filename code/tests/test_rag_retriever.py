"""
tests/test_rag_retriever.py
----------------------------
Unit tests for the RAG retriever module.

These tests require the ChromaDB store to be built first:
    python rag/indexer.py

Run with: python -m pytest tests/test_rag_retriever.py -v
"""

import pytest
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from rag.retriever import retrieve_relevant_clauses, is_store_ready, store_info

STORE_READY = is_store_ready()


# ---------------------------------------------------------------------------
# Store state tests (run regardless of store presence)
# ---------------------------------------------------------------------------

def test_store_info_returns_dict():
    info = store_info()
    assert isinstance(info, dict)
    assert "ready" in info
    assert "chunk_count" in info
    assert "store_path" in info


def test_retrieve_returns_list_even_when_store_missing():
    """Retriever must never raise — returns empty list as fallback."""
    result = retrieve_relevant_clauses("some compliance text", k=5)
    assert isinstance(result, list)


# ---------------------------------------------------------------------------
# Tests that require the store to be built
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not STORE_READY, reason="RAG store not built — run python rag/indexer.py")
def test_store_has_chunks():
    info = store_info()
    assert info["chunk_count"] > 0


@pytest.mark.skipif(not STORE_READY, reason="RAG store not built — run python rag/indexer.py")
def test_retrieve_returns_correct_count():
    results = retrieve_relevant_clauses("AI-generated content transparency disclosure", k=3)
    assert len(results) <= 3
    assert len(results) >= 1


@pytest.mark.skipif(not STORE_READY, reason="RAG store not built — run python rag/indexer.py")
def test_retrieve_result_schema():
    results = retrieve_relevant_clauses("automated decision making personal data", k=3)
    for r in results:
        assert "text" in r
        assert "regulation" in r
        assert "article_id" in r
        assert "reference" in r
        assert "similarity_score" in r
        assert isinstance(r["similarity_score"], float)
        assert 0.0 <= r["similarity_score"] <= 1.0


@pytest.mark.skipif(not STORE_READY, reason="RAG store not built — run python rag/indexer.py")
def test_transparency_query_retrieves_eu_ai_act():
    """A transparency-related query should retrieve EU AI Act Article 50."""
    results = retrieve_relevant_clauses(
        "This AI-generated notice does not disclose that it was created by an AI system."
    )
    regulations = [r["regulation"] for r in results]
    assert any("EU AI Act" in reg for reg in regulations), (
        f"Expected EU AI Act in results, got: {regulations}"
    )


@pytest.mark.skipif(not STORE_READY, reason="RAG store not built — run python rag/indexer.py")
def test_privacy_query_retrieves_gdpr():
    """A personal data query should retrieve GDPR clauses."""
    results = retrieve_relevant_clauses(
        "Dear John Smith, your personal data john@example.com was used in this decision."
    )
    regulations = [r["regulation"] for r in results]
    assert any("GDPR" in reg for reg in regulations), (
        f"Expected GDPR in results, got: {regulations}"
    )


@pytest.mark.skipif(not STORE_READY, reason="RAG store not built — run python rag/indexer.py")
def test_indian_data_query_retrieves_dpdpa():
    """An India-context query should retrieve DPDPA clauses."""
    results = retrieve_relevant_clauses(
        "Processing personal data of Indian residents requires consent under DPDPA."
    )
    regulations = [r["regulation"] for r in results]
    assert any("DPDPA" in reg for reg in regulations), (
        f"Expected DPDPA in results, got: {regulations}"
    )


@pytest.mark.skipif(not STORE_READY, reason="RAG store not built — run python rag/indexer.py")
def test_similarity_scores_are_ordered():
    """Results should be returned in descending similarity order."""
    results = retrieve_relevant_clauses("explainability transparency accountability", k=5)
    scores = [r["similarity_score"] for r in results]
    assert scores == sorted(scores, reverse=True), "Results not sorted by similarity"


@pytest.mark.skipif(not STORE_READY, reason="RAG store not built — run python rag/indexer.py")
def test_empty_string_does_not_crash():
    results = retrieve_relevant_clauses("", k=3)
    assert isinstance(results, list)


@pytest.mark.skipif(not STORE_READY, reason="RAG store not built — run python rag/indexer.py")
def test_k_greater_than_store_size_does_not_crash():
    results = retrieve_relevant_clauses("human oversight review decision", k=1000)
    info = store_info()
    assert len(results) <= info["chunk_count"]
