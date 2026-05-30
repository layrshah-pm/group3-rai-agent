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

Supported formats:
    .pdf  — Extracted via PyMuPDF (fitz) with document-aware chunking
    .txt  — Parsed for [CHUNK]...[/CHUNK] blocks (legacy format, backward-compatible)
"""

import argparse
import hashlib
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

STORE_PATH   = Path(__file__).parent / "policy_store"
DOCS_DIR     = Path(__file__).parent / "documents"
COLLECTION   = "policy_regulations"
EMBED_MODEL  = "all-MiniLM-L6-v2"
MIN_CHUNK_LEN = 80

# ── Legacy .txt format patterns (backward-compatible) ──────────────────────────
CHUNK_PATTERN = re.compile(r'\[CHUNK\](.*?)\[/CHUNK\]', re.DOTALL)
META_PATTERNS = {
    "regulation":      re.compile(r'\[REGULATION:\s*(.+?)\]'),
    "article_id":      re.compile(r'\[ARTICLE:\s*(.+?)\]'),
    "reference":       re.compile(r'\[REFERENCE:\s*(.+?)\]'),
    "pillar_relevance": re.compile(r'\[PILLAR:\s*(.+?)\]'),
}

# ── Pillar relevance mappings ──────────────────────────────────────────────────
_EU_PILLAR: dict[int, str] = {
    5: "fairness", 9: "governance", 10: "fairness",
    13: "transparency", 14: "governance", 17: "governance",
    25: "governance", 50: "transparency", 72: "robustness",
}
_EU_TITLES: dict[int, str] = {
    5:  "Prohibited AI Practices",
    9:  "Risk Management System",
    10: "Data and Data Governance",
    13: "Transparency and Provision of Information to Deployers",
    14: "Human Oversight",
    17: "Quality Management System",
    25: "Responsibilities Along the AI Value Chain",
    50: "Transparency Obligations for Providers and Deployers of Certain AI Systems",
    72: "Post-Market Monitoring by Providers and Post-Market Monitoring Plan",
}

_GDPR_PILLAR: dict[int, str] = {
    5: "privacy", 6: "privacy", 13: "transparency",
    14: "transparency", 17: "privacy", 22: "fairness",
    24: "governance", 32: "robustness",
}

_DPDPA_PILLAR: dict[int, str] = {
    4: "privacy", 6: "privacy", 7: "transparency",
    8: "governance", 11: "transparency", 12: "privacy",
    13: "governance",
}

_NIST_PILLAR: dict[str, str] = {
    "GOVERN 1.1": "governance", "GOVERN 6.1": "governance",
    "MAP 3.5":    "fairness",
    "MEASURE 2.2": "fairness", "MEASURE 2.5": "transparency",
    "MEASURE 2.6": "robustness",
    "MANAGE 1.1": "robustness", "MANAGE 1.3": "governance",
}
_NIST_CATEGORY_PILLAR: dict[str, str] = {
    "GOVERN":  "governance",
    "MAP":     "fairness",
    "MEASURE": "transparency",
    "MANAGE":  "governance",
}
_NIST_CATEGORY_DESC: dict[str, str] = {
    "GOVERN":  "GOVERN Function: Responsible AI risk governance — policies, accountability, and transparency practices across the organisation.",
    "MAP":     "MAP Function: AI risk contextualization — categorization, stakeholder impact assessment, and risk framing.",
    "MEASURE": "MEASURE Function: AI risk analysis and metrics — quantification, assessment, bias measurement, and performance monitoring.",
    "MANAGE":  "MANAGE Function: AI risk treatment — prioritization, response planning, incident management, and ongoing monitoring.",
}


# ── PDF text extraction ────────────────────────────────────────────────────────

def _pdf_to_text(file_path: Path) -> str:
    """Extract full plain text from a PDF using PyMuPDF (fitz)."""
    try:
        import fitz  # type: ignore[import]
        doc = fitz.open(str(file_path))
        pages = [page.get_text() for page in doc]
        doc.close()
        return "\n".join(pages)
    except Exception as exc:
        print(f"  [WARN] PDF extraction failed for {file_path.name}: {exc}")
        return ""


# ── Generic sliding-window helper ─────────────────────────────────────────────

def _sliding_window_chunks(
    text: str, chunk_size: int = 800, overlap: int = 150
) -> list[str]:
    """
    Split text into overlapping character-level windows.
    Tries to break at sentence or paragraph boundaries.
    """
    chunks: list[str] = []
    text_len = len(text)
    start = 0

    while start < text_len:
        end = min(start + chunk_size, text_len)

        if end < text_len:
            # Prefer paragraph break, then sentence boundary
            para_break = text.rfind("\n\n", start, end)
            if para_break != -1 and para_break > start + chunk_size // 2:
                end = para_break + 2
            else:
                line_break = text.rfind("\n", start, end)
                if line_break != -1 and line_break > start + chunk_size // 2:
                    end = line_break + 1
                else:
                    sent_break = text.rfind(". ", start, end)
                    if sent_break != -1 and sent_break > start + chunk_size // 2:
                        end = sent_break + 2

        chunk = text[start:end].strip()
        if len(chunk) >= MIN_CHUNK_LEN:
            chunks.append(chunk)

        step = max(1, end - overlap)
        if step <= start:
            start += 1
        else:
            start = step

    return chunks


def _group_paragraphs(text: str, target: int = 2000, minimum: int = MIN_CHUNK_LEN) -> list[str]:
    """
    Group double-newline-separated paragraphs into non-overlapping chunks
    that are as close to `target` chars as possible without exceeding 3× target.
    Avoids the early-break problem of sliding windows on densely formatted text.
    """
    paras = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    chunks: list[str] = []
    buf = ""

    for para in paras:
        if not buf:
            buf = para
        elif len(buf) + len(para) + 2 <= target * 1.5:
            buf += "\n\n" + para
        else:
            if len(buf) >= minimum:
                chunks.append(buf)
            buf = para

    if buf and len(buf) >= minimum:
        chunks.append(buf)

    return chunks


def _make_chunk(text: str, metadata: dict) -> dict:
    """Build a chunk dict with deterministic MD5 ID."""
    chunk_id = hashlib.md5(text.encode()).hexdigest()[:16]
    return {"id": chunk_id, "text": text, "metadata": metadata}


# ── Document-aware PDF chunkers ────────────────────────────────────────────────

def _chunk_eu_ai_act(text: str, source_file: str) -> list[dict]:
    """
    Article-aware chunking for the EU AI Act (TA-9-2024-0138_EN.pdf).

    The PDF contains "Article N" on its own line in three places: the Table of
    Contents, the running page header (once per page of that article), and the
    article body. We merge ALL text segments for each article number so that
    page-header repetitions are collapsed back into one coherent article chunk.

    Articles <= 1500 merged chars → one chunk.
    Longer articles → 1000-char sliding window / 150-char overlap sub-chunks.
    """
    art_re = re.compile(r"(?m)^[ \t]*(Article\s+(\d+))[ \t]*$")
    matches = list(art_re.finditer(text))

    if not matches:
        print(f"  [WARN] No Article headers found in {source_file} — using sliding window")
        return [
            _make_chunk(c, {
                "regulation": "EU AI Act", "article_id": "General",
                "reference": "EU AI Act — General", "source_file": source_file,
                "pillar_relevance": "general",
            })
            for c in _sliding_window_chunks(text, 1000, 150)
        ]

    # Collect every text segment between consecutive "Article N" occurrences,
    # keyed by article number. Merging handles TOC entries + page-header repeats.
    article_segments: dict[int, list[str]] = {}
    for i, match in enumerate(matches):
        try:
            art_num = int(match.group(2))
        except ValueError:
            continue
        start   = match.end()
        end     = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        segment = text[start:end].strip()
        if segment:
            article_segments.setdefault(art_num, []).append(segment)

    chunks: list[dict] = []

    for art_num in sorted(article_segments.keys()):
        full_text = "\n\n".join(article_segments[art_num])

        if len(full_text) < MIN_CHUNK_LEN:
            continue

        art_label = f"Article {art_num}"
        pillar    = _EU_PILLAR.get(art_num, "general")
        ref_title = _EU_TITLES.get(art_num, art_label)
        reference = f"EU AI Act {art_label} — {ref_title}"
        base_meta = {
            "regulation":      "EU AI Act",
            "article_id":      art_label,
            "reference":       reference,
            "source_file":     source_file,
            "pillar_relevance": pillar,
        }

        if len(full_text) <= 2000:
            chunks.append(_make_chunk(full_text, base_meta))
        else:
            # Paragraph-grouping avoids sliding-window over-fragmentation
            # on the EU AI Act's densely numbered paragraph structure.
            sub_chunks = _group_paragraphs(full_text, target=2000)
            for j, sub in enumerate(sub_chunks):
                if len(sub) < MIN_CHUNK_LEN:
                    continue
                meta = {**base_meta, "article_id": f"{art_label} §{j + 1}"}
                chunks.append(_make_chunk(sub, meta))

    return chunks


def _chunk_gdpr(text: str, source_file: str) -> list[dict]:
    """
    Sliding-window chunking for the GDPR EPSU briefing (GDPR_FINAL_EPSU.pdf).

    800-char chunks / 150-char overlap. Detects first Article reference per chunk
    to assign article_id and pillar_relevance.
    """
    art_ref_re = re.compile(r"\bArticle\s+(\d+)\b", re.IGNORECASE)
    chunks: list[dict] = []

    for chunk_text in _sliding_window_chunks(text, 800, 150):
        if len(chunk_text) < MIN_CHUNK_LEN:
            continue

        art_match = art_ref_re.search(chunk_text)
        if art_match:
            art_num   = int(art_match.group(1))
            art_id    = f"Article {art_num}"
            pillar    = _GDPR_PILLAR.get(art_num, "privacy")
            reference = f"GDPR {art_id}"
        else:
            art_id    = "General"
            pillar    = "privacy"
            reference = "GDPR — General Provisions"

        chunks.append(_make_chunk(chunk_text, {
            "regulation":      "GDPR",
            "article_id":      art_id,
            "reference":       reference,
            "source_file":     source_file,
            "pillar_relevance": pillar,
        }))

    return chunks


def _chunk_dpdpa(text: str, source_file: str) -> list[dict]:
    """
    Section-aware chunking for India DPDPA 2023 (dpdpact.pdf).

    Splits on numbered section headings (e.g. "8. Obligations of Data Fiduciary").
    Short consecutive sections (each < 300 chars) are grouped up to ~900 chars total.
    """
    sec_re = re.compile(r"(?m)^[ \t]*(\d+)\.[ \t]+")
    matches = list(sec_re.finditer(text))

    if not matches:
        print(f"  [WARN] No Section headers found in {source_file} — using sliding window")
        return [
            _make_chunk(c, {
                "regulation": "India DPDPA 2023", "article_id": "General",
                "reference": "DPDPA 2023 — General", "source_file": source_file,
                "pillar_relevance": "privacy",
            })
            for c in _sliding_window_chunks(text, 800, 150)
        ]

    # Extract raw sections
    sections: list[tuple[int, str]] = []
    for i, match in enumerate(matches):
        try:
            sec_num = int(match.group(1))
        except ValueError:
            continue
        start = match.start()
        end   = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections.append((sec_num, text[start:end].strip()))

    # Merge consecutive short sections
    merged: list[tuple[list[int], str]] = []
    buf_nums: list[int] = []
    buf_text = ""

    for sec_num, sec_text in sections:
        if len(sec_text) >= 600:
            # Long section: flush buffer, emit this section alone
            if buf_text:
                merged.append((buf_nums[:], buf_text))
                buf_nums, buf_text = [], ""
            merged.append(([sec_num], sec_text))
        elif buf_text and len(buf_text) + len(sec_text) + 2 > 900:
            merged.append((buf_nums[:], buf_text))
            buf_nums, buf_text = [sec_num], sec_text
        else:
            buf_nums.append(sec_num)
            buf_text = (buf_text + "\n\n" + sec_text).strip() if buf_text else sec_text

    if buf_text:
        merged.append((buf_nums[:], buf_text))

    chunks: list[dict] = []
    for nums, chunk_text in merged:
        if len(chunk_text) < MIN_CHUNK_LEN:
            continue

        primary     = nums[0]
        sec_id      = f"Section {primary}" if len(nums) == 1 else f"Section {nums[0]}–{nums[-1]}"
        pillar      = _DPDPA_PILLAR.get(primary, "privacy")
        reference   = f"DPDPA 2023 {sec_id}"

        chunks.append(_make_chunk(chunk_text, {
            "regulation":      "India DPDPA 2023",
            "article_id":      sec_id,
            "reference":       reference,
            "source_file":     source_file,
            "pillar_relevance": pillar,
        }))

    return chunks


def _chunk_nist(text: str, source_file: str) -> list[dict]:
    """
    Subcategory-aware chunking for NIST AI RMF 1.0 (nist.ai.100-1.pdf).

    Splits on FUNCTION X.Y: patterns (e.g. "GOVERN 1.1:").
    Prepends a parent-category description to each chunk for embedding quality.
    """
    sub_re = re.compile(
        r"(?m)^[ \t]*(GOVERN|MAP|MEASURE|MANAGE)\s+(\d+\.\d+):"
    )
    matches = list(sub_re.finditer(text))

    if not matches:
        print(f"  [WARN] No NIST subcategory headers found in {source_file} — using sliding window")
        return [
            _make_chunk(c, {
                "regulation": "NIST AI RMF 1.0", "article_id": "General",
                "reference": "NIST AI RMF 1.0 — General", "source_file": source_file,
                "pillar_relevance": "governance",
            })
            for c in _sliding_window_chunks(text, 800, 150)
        ]

    chunks: list[dict] = []

    for i, match in enumerate(matches):
        category    = match.group(1)          # e.g. "GOVERN"
        sub_num     = match.group(2)          # e.g. "1.1"
        sub_id      = f"{category} {sub_num}"

        start = match.start()
        end   = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sub_text = text[start:end].strip()

        if len(sub_text) < MIN_CHUNK_LEN:
            continue

        pillar      = _NIST_PILLAR.get(sub_id, _NIST_CATEGORY_PILLAR.get(category, "general"))
        reference   = f"NIST AI RMF 1.0 {sub_id}"
        cat_desc    = _NIST_CATEGORY_DESC.get(category, f"{category} Function")

        # Prepend category context for richer embeddings
        embed_text  = f"{cat_desc}\n\n{sub_text}"

        chunks.append(_make_chunk(embed_text, {
            "regulation":      "NIST AI RMF 1.0",
            "article_id":      sub_id,
            "reference":       reference,
            "source_file":     source_file,
            "pillar_relevance": pillar,
        }))

    return chunks


# ── PDF dispatcher ─────────────────────────────────────────────────────────────

_PDF_CHUNKER_MAP = {
    "TA-9-2024-0138_EN.pdf": _chunk_eu_ai_act,
    "GDPR_FINAL_EPSU.pdf":   _chunk_gdpr,
    "dpdpact.pdf":           _chunk_dpdpa,
    "nist.ai.100-1.pdf":     _chunk_nist,
}


def extract_pdf_chunks(file_path: Path) -> list[dict]:
    """
    Dispatch to the correct document chunker based on filename.
    Unknown PDFs fall back to generic sliding-window chunking.
    Gracefully returns [] on extraction failure.
    """
    filename = file_path.name
    text = _pdf_to_text(file_path)

    if not text.strip():
        print(f"  [WARN] No text extracted from {filename}")
        return []

    print(f"  Extracted {len(text):,} chars from {filename}")

    chunker = _PDF_CHUNKER_MAP.get(filename)
    if chunker is not None:
        return chunker(text, filename)

    # Unknown PDF — generic fallback
    print(f"  [WARN] No specialised chunker for {filename} — using generic sliding window")
    return [
        _make_chunk(c, {
            "regulation":      "Unknown",
            "article_id":      "General",
            "reference":       f"{filename} — General",
            "source_file":     filename,
            "pillar_relevance": "general",
        })
        for c in _sliding_window_chunks(text, 800, 150)
    ]


# ── Legacy .txt parser (backward-compatible) ───────────────────────────────────

def parse_chunks(file_path: Path) -> list[dict]:
    """
    Parses a .txt regulation document for [CHUNK]...[/CHUNK] blocks.
    Returns list of {id, text, metadata} dicts.
    Backward-compatible with the original indexer format.
    """
    raw        = file_path.read_text(encoding="utf-8")
    raw_chunks = CHUNK_PATTERN.findall(raw)
    parsed: list[dict] = []

    meta_prefixes = tuple(f"[{k.upper()}:" for k in META_PATTERNS)

    for chunk_text in raw_chunks:
        chunk_text = chunk_text.strip()
        if not chunk_text:
            continue

        meta: dict[str, str] = {"source_file": file_path.name}
        for key, pattern in META_PATTERNS.items():
            m = pattern.search(chunk_text)
            if m:
                meta[key] = m.group(1).strip()
            elif key == "pillar_relevance":
                meta[key] = "general"   # default so pillar filter queries work
            else:
                meta[key] = "Unknown"

        # Strip metadata header lines to get clean body text
        body_lines = [
            line for line in chunk_text.split("\n")
            if line.strip() and not any(line.strip().startswith(p) for p in meta_prefixes)
        ]
        body = "\n".join(body_lines).strip()

        if len(body) < MIN_CHUNK_LEN:
            continue

        chunk_id = hashlib.md5(body.encode()).hexdigest()[:16]
        parsed.append({"id": chunk_id, "text": body, "metadata": meta})

    return parsed


# ── Store builder ──────────────────────────────────────────────────────────────

def build_store(docs_dir: Path, store_path: Path, reset: bool = False) -> int:
    """
    Indexes all documents in docs_dir into ChromaDB at store_path.
    Processes .pdf files first (document-aware chunking), then .txt files (legacy format).
    Returns the total number of chunks indexed.
    """
    import chromadb
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

    if reset and store_path.exists():
        print(f"[INDEXER] Resetting store at {store_path}...")
        shutil.rmtree(store_path)

    store_path.mkdir(parents=True, exist_ok=True)

    print(f"[INDEXER] Initialising ChromaDB at {store_path}...")
    client   = chromadb.PersistentClient(path=str(store_path))
    embed_fn = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    print(f"[INDEXER] Embedding model: {EMBED_MODEL}")

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

    total = 0

    # ── PDFs ──────────────────────────────────────────────────────────────────
    pdf_files = sorted(docs_dir.glob("*.pdf"))
    if pdf_files:
        print(f"\n[INDEXER] Processing {len(pdf_files)} PDF(s)...")
        for pdf_file in pdf_files:
            try:
                chunks = extract_pdf_chunks(pdf_file)
            except Exception as exc:
                print(f"  [ERROR] Failed to process {pdf_file.name}: {exc}")
                continue

            if not chunks:
                print(f"  {pdf_file.name}: no chunks extracted")
                continue

            # Deduplicate within this file (identical body → same MD5 → keep first)
            seen: set[str] = set()
            chunks = [c for c in chunks if not (c["id"] in seen or seen.add(c["id"]))]  # type: ignore[func-returns-value]

            # Upsert in batches of 100 to avoid oversized requests
            batch_size = 100
            for batch_start in range(0, len(chunks), batch_size):
                batch = chunks[batch_start : batch_start + batch_size]
                collection.upsert(
                    ids=[c["id"] for c in batch],
                    documents=[c["text"] for c in batch],
                    metadatas=[c["metadata"] for c in batch],
                )

            print(f"  {pdf_file.name}: {len(chunks)} chunks indexed")
            total += len(chunks)

    # ── .txt files (legacy format) ────────────────────────────────────────────
    txt_files = sorted(docs_dir.glob("*.txt"))
    if txt_files:
        print(f"\n[INDEXER] Processing {len(txt_files)} .txt file(s)...")
        for txt_file in txt_files:
            chunks = parse_chunks(txt_file)
            if not chunks:
                print(f"  {txt_file.name}: no chunks found — check [CHUNK]...[/CHUNK] format")
                continue

            collection.upsert(
                ids=[c["id"] for c in chunks],
                documents=[c["text"] for c in chunks],
                metadatas=[c["metadata"] for c in chunks],
            )
            print(f"  {txt_file.name}: {len(chunks)} chunks indexed")
            total += len(chunks)

    if not pdf_files and not txt_files:
        print(f"[INDEXER] No documents found in {docs_dir}")
        return 0

    print(f"\n[INDEXER] Done. Total chunks: {total}")
    return total


# ── Store verifier ─────────────────────────────────────────────────────────────

def verify_store(store_path: Path) -> None:
    """Prints store stats and sample retrieval results."""
    import chromadb
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

    if not store_path.exists() or not (store_path / "chroma.sqlite3").exists():
        print(f"[INDEXER] Store not found at {store_path}. Run indexer first.")
        return

    client   = chromadb.PersistentClient(path=str(store_path))
    embed_fn = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)

    try:
        collection = client.get_collection(COLLECTION, embedding_function=embed_fn)
    except Exception as exc:
        print(f"[INDEXER] Could not open collection: {exc}")
        return

    count = collection.count()
    print(f"\n[INDEXER] Store: {store_path}")
    print(f"[INDEXER] Collection: {COLLECTION}")
    print(f"[INDEXER] Total chunks: {count}")

    # Sample by regulation
    try:
        sample = collection.get(limit=200, include=["metadatas"])
        reg_counts: dict[str, int] = {}
        for meta in sample["metadatas"]:
            reg = meta.get("regulation", "Unknown")
            reg_counts[reg] = reg_counts.get(reg, 0) + 1
        print("\n[INDEXER] Chunks per regulation (sample of up to 200):")
        for reg, cnt in sorted(reg_counts.items()):
            print(f"  {reg}: {cnt}")
    except Exception:
        pass

    _run_sample_query(
        collection,
        "AI-generated content must be disclosed to users",
    )
    _run_sample_query(
        collection,
        "personal data privacy rights of Indian residents",
    )
    _run_sample_query(
        collection,
        "bias fairness disparate impact model evaluation",
    )


def _run_sample_query(collection, query: str) -> None:
    print(f'\n[INDEXER] Sample: "{query}"')
    try:
        results = collection.query(
            query_texts=[query],
            n_results=min(3, collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            sim = round(max(0.0, 1.0 - dist / 2.0), 3)
            print(f"  [{meta.get('regulation')} — {meta.get('article_id')}] sim={sim}")
            print(f"  {doc[:200].replace(chr(10), ' ')}...")
    except Exception as exc:
        print(f"  [WARN] Query failed: {exc}")


# ── CLI entry point ────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Index regulatory documents (PDF + TXT) into ChromaDB."
    )
    parser.add_argument("--reset",    action="store_true", help="Delete and rebuild store from scratch")
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
