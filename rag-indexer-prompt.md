# Claude Code Prompt — RAG Indexer: PDF-Native Chunking for Regulatory Documents

## Context

The RAG system is in `rai-compliance-agent/rag/`. It has:
- `indexer.py` — currently reads only `.txt` files with `[CHUNK]...[/CHUNK]` markers. **No PDFs are indexed.**
- `retriever.py` — runtime ChromaDB retrieval, works fine, keep as-is
- `policy_store/` — ChromaDB vector store directory
- `documents/` — contains the actual regulatory PDFs:
  - `TA-9-2024-0138_EN.pdf` — EU AI Act (459 pages, ~592k chars)
  - `GDPR_FINAL_EPSU.pdf` — GDPR briefing (40 pages, ~68k chars)
  - `dpdpact.pdf` — India DPDPA 2023 (21 pages, ~63k chars)
  - `nist.ai.100-1.pdf` — NIST AI RMF 1.0 (48 pages, ~106k chars)

**The problem:** The indexer only parses `.txt` files. The PDFs are never read. The ChromaDB store is either empty or has only manually created chunks. We need to rebuild `indexer.py` to extract, chunk, and index all 4 PDFs intelligently.

---

## What to Build

Rewrite `rag/indexer.py` to:
1. Read all 4 PDFs using `pymupdf` (already available as `fitz`)
2. Chunk each document using **document-aware strategies** (described below)
3. Attach structured metadata to each chunk: `regulation`, `article_id`, `reference`, `pillar_relevance`
4. Upsert all chunks into ChromaDB using the existing `all-MiniLM-L6-v2` embedding model and `policy_regulations` collection
5. Keep backward compatibility — still process `.txt` files if present

---

## Chunking Strategy Per Document

### 1. EU AI Act (`TA-9-2024-0138_EN.pdf`)

**Structure:** The document contains numbered Articles (`Article 5`, `Article 9`, etc.). Each article has a title and numbered paragraphs.

**Strategy:** Split by Article. Use regex to detect `Article \d+` as a section boundary. Each article = one chunk. For very long articles (>1500 chars), split by paragraph (numbered `1.`, `2.`, etc.) and group into chunks of ~800–1200 chars with a 150-char overlap.

**Priority articles to index** (these are the ones our policy agent cites — index ALL articles but tag these as high priority in metadata):
- Article 5 — Prohibited AI Practices
- Article 9 — Risk Management System
- Article 10 — Data and Data Governance
- Article 13 — Transparency and Provision of Information
- Article 14 — Human Oversight
- Article 17 — Quality Management System
- Article 25 — Responsibilities Along the AI Value Chain
- Article 50 — Transparency Obligations
- Article 72 — Post-Market Monitoring

**Metadata format:**
```python
{
    "regulation": "EU AI Act",
    "article_id": "Article 13",
    "reference": "EU AI Act Article 13 — Transparency and provision of information to deployers",
    "source_file": "TA-9-2024-0138_EN.pdf",
    "pillar_relevance": "transparency"  # see pillar mapping below
}
```

### 2. GDPR (`GDPR_FINAL_EPSU.pdf`)

**Structure:** 40-page EPSU briefing document. Uses section headers and numbered articles referenced inline. Less strictly structured than the other docs.

**Strategy:** Sliding window chunking — 800-char chunks with 150-char overlap. Detect article references in text (e.g. `Article 5`, `Article 17`) and use the first article reference found in each chunk as `article_id`. If none, use the closest preceding section header as `article_id`.

**Metadata format:**
```python
{
    "regulation": "GDPR",
    "article_id": "Article 5",  # first article reference in chunk, or "General"
    "reference": "GDPR Article 5 — Principles relating to processing of personal data",
    "source_file": "GDPR_FINAL_EPSU.pdf",
    "pillar_relevance": "privacy"
}
```

### 3. DPDPA 2023 (`dpdpact.pdf`)

**Structure:** Indian legislation. Uses numbered Sections (`4.`, `6.`, `8.`, etc.) and Chapters. Sections start with a number followed by a period at the beginning of a line.

**Strategy:** Split by Section number. Regex: `^\s*(\d+)\.\s` at line start. Each Section = one chunk. Sections are typically 200–600 chars — group consecutive short sections (< 300 chars each) together to avoid micro-chunks, up to ~900 chars total.

**Key sections to tag with pillar relevance:**
- Section 4 — lawful purpose → `privacy`
- Section 6 — consent → `privacy`
- Section 7 — notice → `transparency`
- Section 8 — data fiduciary obligations → `governance`
- Section 11 — right to information → `transparency`
- Section 12 — correction and erasure → `privacy`
- Section 13 — grievance redressal → `governance`

**Metadata format:**
```python
{
    "regulation": "India DPDPA 2023",
    "article_id": "Section 6",
    "reference": "DPDPA 2023 Section 6 — Consent",
    "source_file": "dpdpact.pdf",
    "pillar_relevance": "privacy"
}
```

### 4. NIST AI RMF 1.0 (`nist.ai.100-1.pdf`)

**Structure:** Uses a function/subcategory system: `GOVERN`, `MAP`, `MEASURE`, `MANAGE`. Subcategories follow the pattern `GOVERN 1.1:`, `MEASURE 2.5:`, etc.

**Strategy:** Split by subcategory. Regex: `([A-Z]+\s+\d+\.\d+):` marks the start of each subcategory entry. Each subcategory = one chunk. Descriptive context (the parent category header) should be prepended to each chunk for embedding quality.

**Key subcategories to tag:**
- `GOVERN 1.1` → `governance`
- `GOVERN 6.1` → `governance`
- `MAP 3.5` → `fairness`
- `MEASURE 2.2` → `fairness`
- `MEASURE 2.5` → `transparency`
- `MEASURE 2.6` → `robustness`
- `MANAGE 1.3` → `governance`

**Metadata format:**
```python
{
    "regulation": "NIST AI RMF 1.0",
    "article_id": "GOVERN 1.1",
    "reference": "NIST AI RMF 1.0 GOVERN 1.1 — Legal and regulatory requirements involving AI",
    "source_file": "nist.ai.100-1.pdf",
    "pillar_relevance": "governance"
}
```

---

## Pillar Relevance Mapping

Use this mapping to tag chunks with `pillar_relevance`. A chunk can only have one primary pillar tag (pick the most relevant):

| Pillar | Tag value | EU AI Act articles | DPDPA sections | NIST subcategories | GDPR articles |
|--------|-----------|-------------------|----------------|-------------------|---------------|
| Governance & Accountability | `governance` | 9, 17, 25 | 8, 13 | GOVERN 1.1, 6.1, MANAGE 1.3 | Art 5, 24 |
| Fairness & Bias Mitigation | `fairness` | 5, 10 | 8 | MAP 3.5, MEASURE 2.2 | Art 22 |
| Transparency & Explainability | `transparency` | 13, 50 | 7, 11 | MEASURE 2.5 | Art 13, 14 |
| Robustness & Monitoring | `robustness` | 72 | — | MEASURE 2.6, MANAGE 1.1 | Art 32 |
| Privacy & Data Stewardship | `privacy` | — | 4, 6, 12 | GOVERN 6.1 | Art 5, 6, 17 |

For chunks that don't clearly map to a pillar, use `"general"`.

---

## Full Rewritten `indexer.py`

Replace the entire file. Keep:
- `STORE_PATH`, `DOCS_DIR`, `COLLECTION`, `EMBED_MODEL` constants
- `build_store()` signature and return value (total chunk count)
- `verify_store()` function
- `main()` with `--reset`, `--verify`, `--docs-dir` args

Add:
- `extract_pdf_chunks(file_path: Path) -> list[dict]` — dispatcher that calls the right parser based on filename
- `_chunk_eu_ai_act(text: str) -> list[dict]`
- `_chunk_gdpr(text: str) -> list[dict]`
- `_chunk_dpdpa(text: str) -> list[dict]`
- `_chunk_nist(text: str) -> list[dict]`
- `_sliding_window_chunks(text: str, chunk_size: int, overlap: int) -> list[str]` — generic helper

Update `build_store()` to process both `.pdf` files (via `extract_pdf_chunks`) and `.txt` files (via existing `parse_chunks`).

Use `fitz` (pymupdf) for PDF text extraction:
```python
import fitz  # pymupdf

def _pdf_to_text(file_path: Path) -> str:
    doc = fitz.open(str(file_path))
    return "\n".join(page.get_text() for page in doc)
```

---

## Expected Output After Running

```
[INDEXER] Initialising ChromaDB at rag/policy_store...
[INDEXER] Embedding model: all-MiniLM-L6-v2
[INDEXER] Processing PDFs...
  TA-9-2024-0138_EN.pdf (EU AI Act): ~180–220 chunks indexed
  GDPR_FINAL_EPSU.pdf (GDPR):         ~80–100 chunks indexed
  dpdpact.pdf (DPDPA 2023):           ~40–60 chunks indexed
  nist.ai.100-1.pdf (NIST AI RMF):    ~60–80 chunks indexed
[INDEXER] Done. Total chunks: ~360–460
```

---

## Also Update `retriever.py` — Add Pillar-Filtered Retrieval

After rewriting the indexer, add one new function to `retriever.py`:

```python
def retrieve_clauses_for_pillar(pillar_tag: str, k: int = 5) -> list[dict]:
    """
    Retrieves regulatory clauses tagged for a specific RAI pillar.
    Used by the policy document auditor to fetch grounding clauses per pillar.
    
    pillar_tag: one of "governance", "fairness", "transparency", "robustness", "privacy"
    """
```

This uses ChromaDB's `where` filter on the `pillar_relevance` metadata field:
```python
results = collection.query(
    query_texts=[f"AI policy requirements for {pillar_tag}"],
    n_results=k,
    where={"pillar_relevance": {"$in": [pillar_tag, "general"]}},
    include=["documents", "metadatas", "distances"],
)
```

This function is what the new `policy_agent.py` policy document mode calls — it fetches the relevant regulatory clauses for each pillar before asking Gemma to evaluate the uploaded policy document.

---

## RBI MRM (Not in PDFs)

The RBI Model Risk Management Guidelines (2023) are not in the documents folder as a PDF. Handle this by creating a small hardcoded text snippet file at `rag/documents/rbi_mrm_key_clauses.txt` using the **existing** `[CHUNK]...[/CHUNK]` format the current indexer already supports. Include 5–8 key clauses covering:

- Section 3.2 — Model validation methodology
- Section 4.1 — Model inventory and named ownership
- Section 7 — Fairness and bias considerations in models
- Section 8 — Ongoing monitoring and drift detection

Format each as:
```
[CHUNK]
[REGULATION: RBI Model Risk Management Guidelines 2023]
[ARTICLE: Section 4.1]
[REFERENCE: RBI MRM 2023 Section 4.1 — Model Inventory and Named Accountability]
Every AI/ML model used in financial or operational decisions must be registered in a 
central model inventory. Each model must have a named individual owner responsible 
for its governance, performance, and compliance. The inventory must be maintained 
current and reviewed quarterly.
[/CHUNK]
```

---

## Constraints

- Use only `fitz` (pymupdf) for PDF reading — already installed in the venv
- Do not change the ChromaDB collection name (`policy_regulations`) or embedding model (`all-MiniLM-L6-v2`)
- Keep chunk IDs deterministic (MD5 hash of body text) — existing behaviour
- Keep the `--reset` flag working so the store can be fully rebuilt
- Minimum chunk body length: 80 chars (skip shorter fragments)
- Print progress to console during indexing so it's clear what's happening
