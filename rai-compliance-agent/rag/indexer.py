"""
rag/indexer.py
--------------
Indexes regulatory documents into ChromaDB for the RAG policy agent.

Run this once before using the RAG-powered policy agent:
    python rag/indexer.py

Options:
    --reset     Delete and rebuild the store from scratch
    --verify    Print chunk count and sample entries without rebuilding
    --docs-dir  Path to documents directory (default: rag/documents/)

Each .txt file in docs-dir is parsed for [CHUNK]...[/CHUNK] blocks.
Each chunk is embedded with all-MiniLM-L6-v2 and stored in ChromaDB.
"""

import argparse
import hashlib
import re
import shutil
import sys
from pathlib import Path

# Ensure project root is on path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

STORE_PATH   = Path(__file__).parent / "policy_store"
DOCS_DIR     = Path(__file__).parent / "documents"
COLLECTION   = "policy_regulations"
EMBED_MODEL  = "all-MiniLM-L6-v2"

CHUNK_PATTERN = re.compile(
    r'\[CHUNK\](.*?)\[/CHUNK\]',
    re.DOTALL,
)
META_PATTERNS = {
    "regulation": re.compile(r'\[REGULATION:\s*(.+?)\]'),
    "article_id": re.compile(r'\[ARTICLE:\s*(.+?)\]'),
    "reference":  re.compile(r'\[REFERENCE:\s*(.+?)\]'),
}


def parse_chunks(file_path: Path) -> list[dict]:
    """
    Parses a regulation document into chunks.
    Returns list of {id, text, metadata} dicts.
    """
    raw = file_path.read_text(encoding="utf-8")
    raw_chunks = CHUNK_PATTERN.findall(raw)

    parsed = []
    for chunk_text in raw_chunks:
        chunk_text = chunk_text.strip()
        if not chunk_text:
            continue

        # Extract metadata from header lines
        meta = {"source_file": file_path.name}
        for key, pattern in META_PATTERNS.items():
            match = pattern.search(chunk_text)
            meta[key] = match.group(1).strip() if match else "Unknown"

        # Strip metadata header lines from the body text
        body_lines = []
        for line in chunk_text.split("\n"):
            stripped = line.strip()
            is_meta = any(stripped.startswith(f"[{k.upper()}:") for k in META_PATTERNS)
            if not is_meta and stripped:
                body_lines.append(line)
        body = "\n".join(body_lines).strip()

        if len(body) < 50:
            continue  # skip near-empty chunks

        # Deterministic ID from content hash
        chunk_id = hashlib.md5(body.encode()).hexdigest()[:16]

        parsed.append({
            "id":       chunk_id,
            "text":     body,
            "metadata": meta,
        })

    return parsed


def build_store(docs_dir: Path, store_path: Path, reset: bool = False) -> int:
    """
    Indexes all documents in docs_dir into ChromaDB at store_path.
    Returns the total number of chunks indexed.
    """
    import chromadb
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

    if reset and store_path.exists():
        print(f"[INDEXER] Resetting store at {store_path}...")
        shutil.rmtree(store_path)

    store_path.mkdir(parents=True, exist_ok=True)

    print(f"[INDEXER] Initialising ChromaDB at {store_path}...")
    client = chromadb.PersistentClient(path=str(store_path))

    embed_fn = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    print(f"[INDEXER] Embedding model: {EMBED_MODEL}")

    # Get or create collection
    try:
        collection = client.get_collection(COLLECTION, embedding_function=embed_fn)
        print(f"[INDEXER] Found existing collection with {collection.count()} chunks.")
    except Exception:
        collection = client.create_collection(
            name=COLLECTION,
            embedding_function=embed_fn,
            metadata={"hnsw:space": "cosine"},
        )
        print(f"[INDEXER] Created new collection: {COLLECTION}")

    # Parse and index all .txt files
    doc_files = list(docs_dir.glob("*.txt"))
    if not doc_files:
        print(f"[INDEXER] No .txt files found in {docs_dir}")
        return 0

    total = 0
    for doc_file in sorted(doc_files):
        chunks = parse_chunks(doc_file)
        if not chunks:
            print(f"  {doc_file.name}: no chunks found — check [CHUNK]...[/CHUNK] format")
            continue

        # Upsert (add or update) to avoid duplicates on re-run
        collection.upsert(
            ids=[c["id"] for c in chunks],
            documents=[c["text"] for c in chunks],
            metadatas=[c["metadata"] for c in chunks],
        )
        print(f"  {doc_file.name}: {len(chunks)} chunks indexed")
        total += len(chunks)

    print(f"\n[INDEXER] Done. Total chunks: {total}")
    return total


def verify_store(store_path: Path) -> None:
    """Prints store stats and sample entries."""
    import chromadb
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

    if not store_path.exists():
        print(f"[INDEXER] Store not found at {store_path}. Run indexer first.")
        return

    client = chromadb.PersistentClient(path=str(store_path))
    embed_fn = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)

    try:
        collection = client.get_collection(COLLECTION, embedding_function=embed_fn)
    except Exception as e:
        print(f"[INDEXER] Could not open collection: {e}")
        return

    count = collection.count()
    print(f"\n[INDEXER] Store: {store_path}")
    print(f"[INDEXER] Collection: {COLLECTION}")
    print(f"[INDEXER] Total chunks: {count}")

    print("\n[INDEXER] Sample retrieval for query: 'AI-generated content must be disclosed'")
    results = collection.query(
        query_texts=["AI-generated content must be disclosed"],
        n_results=3,
        include=["documents", "metadatas", "distances"],
    )
    for i, (doc, meta, dist) in enumerate(zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    )):
        sim = round(max(0.0, 1.0 - dist / 2.0), 3)
        print(f"\n  [{i+1}] {meta.get('regulation')} — {meta.get('article_id')}")
        print(f"       Similarity: {sim}")
        print(f"       {doc[:200]}...")

    print("\n[INDEXER] Sample retrieval for query: 'personal data privacy Indian residents'")
    results2 = collection.query(
        query_texts=["personal data privacy Indian residents"],
        n_results=3,
        include=["documents", "metadatas", "distances"],
    )
    for i, (doc, meta, dist) in enumerate(zip(
        results2["documents"][0],
        results2["metadatas"][0],
        results2["distances"][0],
    )):
        sim = round(max(0.0, 1.0 - dist / 2.0), 3)
        print(f"\n  [{i+1}] {meta.get('regulation')} — {meta.get('article_id')}")
        print(f"       Similarity: {sim}")
        print(f"       {doc[:200]}...")


def main():
    parser = argparse.ArgumentParser(description="Index regulatory documents into ChromaDB.")
    parser.add_argument("--reset",    action="store_true", help="Delete and rebuild store")
    parser.add_argument("--verify",   action="store_true", help="Verify store without rebuilding")
    parser.add_argument("--docs-dir", type=Path, default=DOCS_DIR, help="Path to documents directory")
    args = parser.parse_args()

    if args.verify:
        verify_store(STORE_PATH)
        return

    total = build_store(args.docs_dir, STORE_PATH, reset=args.reset)
    if total > 0:
        print("\nRun with --verify to inspect the store.")


if __name__ == "__main__":
    main()
