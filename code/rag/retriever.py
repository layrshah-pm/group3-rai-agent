"""
rag/retriever.py
----------------
Runtime retrieval helper for the RAG-powered Policy Agent.

Wraps ChromaDB to retrieve the most semantically relevant regulatory
clauses for a given text. Used by policy_agent.py to augment the
compliance check prompt with actual regulatory language.

Falls back silently if the store is not yet indexed (no crash).

Usage:
    from rag.retriever import retrieve_relevant_clauses

    clauses = retrieve_relevant_clauses(text, k=5)
    # Returns: list of dicts with keys: text, regulation, article_id, reference, similarity_score
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_STORE_PATH = Path(__file__).parent / "policy_store"
_COLLECTION_NAME = "policy_regulations"
_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Clauses with cosine similarity below this threshold are not sent to the LLM.
# Irrelevant clauses at low similarity can mislead the model more than help it.
MIN_SIMILARITY = 0.25

# Module-level singletons — initialised once on first use
_client = None
_collection = None
_embedding_fn = None


def _get_embedding_fn():
    """Lazy-load the sentence-transformer embedding function."""
    global _embedding_fn
    if _embedding_fn is None:
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
        _embedding_fn = SentenceTransformerEmbeddingFunction(model_name=_EMBEDDING_MODEL)
    return _embedding_fn


def _get_collection():
    """
    Lazy-load the ChromaDB collection.
    Returns None if the store does not exist yet (not indexed).
    """
    global _client, _collection

    if _collection is not None:
        return _collection

    # Check for ChromaDB data (chroma.sqlite3) — not just .gitkeep
    if not _STORE_PATH.exists() or not (_STORE_PATH / "chroma.sqlite3").exists():
        logger.warning(
            "[RAG] policy_store not found at %s. "
            "Run `python rag/indexer.py` to build the store. "
            "Falling back to hardcoded criteria only.",
            _STORE_PATH,
        )
        return None

    try:
        import chromadb
        _client = chromadb.PersistentClient(path=str(_STORE_PATH))
        _collection = _client.get_collection(
            name=_COLLECTION_NAME,
            embedding_function=_get_embedding_fn(),
        )
        count = _collection.count()
        logger.info("[RAG] Connected to policy_store — %d chunks indexed.", count)
        return _collection
    except Exception as e:
        logger.warning("[RAG] Could not load policy_store: %s. Falling back.", e)
        return None


def retrieve_relevant_clauses(text: str, k: int = 5) -> list[dict]:
    """
    Retrieves the k most semantically relevant regulatory clauses for the
    given text. Uses cosine similarity via ChromaDB + sentence-transformers.

    Args:
        text: The AI-generated text being audited (current_text from state).
        k:    Number of top clauses to retrieve. Default 5.

    Returns:
        List of dicts, each with:
            - text:             Full regulatory clause text
            - regulation:       Source regulation name (e.g. "EU AI Act")
            - article_id:       Article identifier (e.g. "Article 13")
            - reference:        Full reference string
            - similarity_score: Cosine similarity 0-1 (1 = most similar)

        Returns empty list if store unavailable or query fails.
    """
    collection = _get_collection()
    if collection is None:
        return []

    try:
        results = collection.query(
            query_texts=[text],
            n_results=min(k, collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        clauses = []
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        raw_count = len(documents)
        for doc, meta, dist in zip(documents, metadatas, distances):
            # ChromaDB returns L2 distance for cosine space; convert to similarity.
            # With normalised embeddings: cosine_similarity = 1 - (dist / 2)
            similarity = round(max(0.0, 1.0 - dist / 2.0), 4)

            if similarity < MIN_SIMILARITY:
                continue   # discard low-relevance clauses silently accumulated below threshold

            clauses.append({
                "text": doc,
                "regulation": meta.get("regulation", "Unknown"),
                "article_id": meta.get("article_id", "Unknown"),
                "reference": meta.get("reference", ""),
                "similarity_score": similarity,
            })

        filtered_out = raw_count - len(clauses)
        if filtered_out > 0:
            logger.info(
                "[RAG] %d/%d clause(s) filtered out (similarity < %.2f).",
                filtered_out, raw_count, MIN_SIMILARITY,
            )

        return clauses

    except Exception as e:
        logger.warning("[RAG] Retrieval failed: %s. Returning empty.", e)
        return []


def retrieve_clauses_for_pillar(pillar_tag: str, k: int = 5) -> list[dict]:
    """
    Retrieves regulatory clauses tagged for a specific RAI pillar.
    Used by the policy document auditor to fetch grounding clauses per pillar.

    pillar_tag: one of "governance", "fairness", "transparency", "robustness", "privacy"
    Returns same dict schema as retrieve_relevant_clauses (text, regulation, article_id,
    reference, similarity_score), plus pillar_relevance from metadata.
    Falls back to empty list if store unavailable or pillar filter matches nothing.
    """
    collection = _get_collection()
    if collection is None:
        return []

    total = collection.count()
    if total == 0:
        return []

    try:
        results = collection.query(
            query_texts=[f"AI policy requirements for {pillar_tag}"],
            n_results=min(k, total),
            where={"pillar_relevance": {"$in": [pillar_tag, "general"]}},
            include=["documents", "metadatas", "distances"],
        )
    except Exception:
        # ChromaDB where-filter can error when no documents match the pillar tag
        # — retry without the filter but log clearly that result quality may be degraded.
        logger.warning(
            "[RAG] pillar filter query failed for '%s': no documents matched the pillar tag. "
            "Retrying without filter — results may include clauses from unrelated pillars.",
            pillar_tag,
        )
        try:
            results = collection.query(
                query_texts=[f"AI policy requirements for {pillar_tag}"],
                n_results=min(k, total),
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            logger.warning("[RAG] retrieve_clauses_for_pillar failed entirely: %s", exc)
            return []

    clauses: list[dict] = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        similarity = round(max(0.0, 1.0 - dist / 2.0), 4)
        if similarity < MIN_SIMILARITY:
            continue
        clauses.append({
            "text":             doc,
            "regulation":       meta.get("regulation", "Unknown"),
            "article_id":       meta.get("article_id", "Unknown"),
            "reference":        meta.get("reference", ""),
            "pillar_relevance": meta.get("pillar_relevance", "general"),
            "similarity_score": similarity,
        })

    return clauses


def is_store_ready() -> bool:
    """Returns True if the vector store exists and has indexed chunks."""
    return _get_collection() is not None


def store_info() -> dict:
    """Returns metadata about the current store state. Used in UI and tests."""
    collection = _get_collection()
    if collection is None:
        return {"ready": False, "chunk_count": 0, "store_path": str(_STORE_PATH)}
    return {
        "ready": True,
        "chunk_count": collection.count(),
        "store_path": str(_STORE_PATH),
    }
