# RAG Implementation Plan — Policy Agent Regulatory Retrieval

**Project:** RAI Compliance Agent — IIMA Capstone EPAIBBL01 Group 3  
**Feature:** RAG-powered Policy Compliance Agent  
**Approach:** Hybrid — hardcoded criteria (reliability) + retrieved regulatory clauses (accuracy + citability)

---

## Overview

The policy agent currently checks text against 11 hardcoded criteria written as one-line descriptions. This works but has two weaknesses a panel will notice: the criteria are summaries, not actual regulation text, and adding a new regulation requires code changes.

This plan upgrades the policy agent to a RAG-powered system. At runtime, it embeds the incoming text, retrieves the most semantically relevant regulatory clauses from a ChromaDB vector store, and passes those retrieved clauses alongside the hardcoded criteria to the LLM. The result: every violation is grounded in actual article language, retrieved clauses appear in the audit trail, and new regulations can be added by dropping a text file into `rag/documents/` and re-running the indexer.

**Infrastructure already in place:** `chromadb>=0.5.0` and `sentence-transformers>=3.0.0` are both declared in `requirements.txt` but unused. No new dependencies needed.

---

## Design Decisions

**Hybrid approach, not pure RAG.** The 11 hardcoded criteria remain as the primary check layer. Retrieved clauses are passed as supplementary context. This matters for demo reliability — if retrieval misses a relevant clause, the hardcoded criteria catch the violation anyway. For a PoC demo you need deterministic, trustworthy behaviour. Pure RAG over regulatory text can miss things.

**Embed the full text, not the criteria.** The ChromaDB query uses `current_text` as the query, not the criteria names. This surfaces regulatory clauses that are topically relevant to what the text is actually about — a loan denial notice retrieves Articles 13 and 22 (explainability, automated decisions), not generic governance articles.

**Graceful fallback.** If the vector store does not exist (user hasn't run the indexer), the policy agent falls back to the current hardcoded-only behaviour with no crash. One print warning. This is critical — do not break the existing demo.

**Store retrieved clauses in state.** Add `retrieved_clauses` to `ComplianceState` so the audit trail records what regulatory text was consulted at each run. This directly addresses the EU AI Act Article 50 requirement for traceable audit records.

**Embedding model:** `all-MiniLM-L6-v2` from sentence-transformers. 384-dimension embeddings, fast, runs locally with no API calls, already in requirements.

---

## Directory Structure After Implementation

```
rai-compliance-agent/
├── rag/
│   ├── __init__.py
│   ├── indexer.py              # One-time document indexing script
│   ├── retriever.py            # Runtime retrieval helper used by policy_agent
│   ├── documents/
│   │   ├── eu_ai_act.txt       # EU AI Act — Articles 5, 8, 9, 10, 13, 14, 15, 50
│   │   ├── gdpr.txt            # GDPR — Articles 5, 13, 22
│   │   ├── nist_ai_rmf.txt     # NIST AI RMF — GOVERN, MAP, MEASURE, MANAGE controls
│   │   ├── dpdpa_2023.txt      # India DPDPA 2023 — Sections 4, 6, 8, 11
│   │   └── rbi_guidelines.txt  # RBI Model Risk Management Guidelines 2023
│   └── policy_store/           # ChromaDB persistent store (created by indexer)
│       └── .gitkeep
├── nodes/
│   └── policy_agent.py         # MODIFIED — RAG retrieval integrated
├── state.py                    # MODIFIED — add retrieved_clauses field
├── ui/
│   └── app.py                  # MODIFIED — show retrieved clauses expander
└── tests/
    └── test_rag_retriever.py   # NEW — retriever unit tests
```

---

## Step 1 — Regulatory Document Files

Create each file below exactly as specified. The content is structured with `[CHUNK]` delimiters that the indexer parses. Each chunk becomes one vector store entry.

### `rag/documents/eu_ai_act.txt`

```
[CHUNK]
[REGULATION: EU AI Act]
[ARTICLE: Article 5 — Prohibited AI Practices]
[REFERENCE: EU AI Act Article 5(1)(b)]

Article 5(1)(b) prohibits the placing on the market, putting into service or use of AI systems that exploit any of the vulnerabilities of a specific group of persons due to their age, disability or a specific social or economic situation in a way that distorts the behaviour of that person, causing or likely to cause that person or another person physical or psychological harm. AI systems that exploit vulnerabilities of protected groups to influence decisions in ways harmful to those individuals are explicitly prohibited.

[/CHUNK]

[CHUNK]
[REGULATION: EU AI Act]
[ARTICLE: Article 9 — Risk Management System]
[REFERENCE: EU AI Act Article 9]

Article 9 requires providers of high-risk AI systems to establish, implement, document and maintain a risk management system. The risk management system shall consist of a continuous iterative process run throughout the entire lifecycle of a high-risk AI system. It shall comprise: identification and analysis of known and foreseeable risks; estimation and evaluation of risks that may emerge when used as intended; evaluation of risks based on data gathered from post-market monitoring. Risk management measures shall ensure that residual risks are judged acceptable.

[/CHUNK]

[CHUNK]
[REGULATION: EU AI Act]
[ARTICLE: Article 10 — Data and Data Governance]
[REFERENCE: EU AI Act Article 10]

Article 10 requires that high-risk AI systems be trained, validated and tested on data sets that are subject to data governance and management practices. Training, validation and testing data sets shall be relevant, sufficiently representative, and to the best extent possible, free of errors and complete in view of the intended purpose. Data sets shall have appropriate statistical properties including where applicable as regards the persons or groups of persons on which the high-risk AI system is intended to be used. Data sets shall take into account the characteristics or elements that are particular to the specific geographical, contextual, behavioural or functional setting. Providers must acknowledge known data limitations.

[/CHUNK]

[CHUNK]
[REGULATION: EU AI Act]
[ARTICLE: Article 13 — Transparency and Provision of Information to Deployers]
[REFERENCE: EU AI Act Article 13]

Article 13 requires that high-risk AI systems be designed and developed in such a way to ensure that their operation is sufficiently transparent to enable deployers to understand and correctly use the system. High-risk AI systems shall be accompanied by instructions for use in an appropriate digital format that include: the identity and contact details of the provider; the characteristics, capabilities and limitations of the system; information on the level of accuracy, robustness and cybersecurity; the intended purpose; the human oversight measures including technical measures to facilitate interpretation of outputs. Affected individuals have the right to receive a meaningful explanation of any decision that significantly affects them.

[/CHUNK]

[CHUNK]
[REGULATION: EU AI Act]
[ARTICLE: Article 14 — Human Oversight]
[REFERENCE: EU AI Act Article 14]

Article 14 requires that high-risk AI systems be designed and developed in such a way that they can be effectively overseen by natural persons during the period they are in use. High-risk AI systems shall be provided with appropriate human-machine interface tools to enable effective human oversight. Human oversight measures shall enable persons to whom human oversight is assigned to: fully understand the capacities and limitations of the high-risk AI system; monitor the operation of the system and detect anomalies, dysfunctions and unexpected performance; intervene and stop the system or override its output. Deployers must implement mechanisms allowing individuals to request human review of automated decisions.

[/CHUNK]

[CHUNK]
[REGULATION: EU AI Act]
[ARTICLE: Article 15 — Accuracy, Robustness and Cybersecurity]
[REFERENCE: EU AI Act Article 15]

Article 15 requires that high-risk AI systems be designed and developed in such a way that they achieve an appropriate level of accuracy, robustness and cybersecurity. The levels of accuracy and relevant accuracy metrics shall be declared in the accompanying instructions of use. High-risk AI systems shall be resilient to errors, faults or inconsistencies that may occur within the system or in the environment in which the system operates. Technical robustness requires that high-risk AI systems identify and communicate uncertainty in predictions or decisions. Providers must communicate confidence scores or uncertainty estimates when the system cannot determine outcomes with sufficient certainty.

[/CHUNK]

[CHUNK]
[REGULATION: EU AI Act]
[ARTICLE: Article 25 — Responsibilities Along the AI Value Chain]
[REFERENCE: EU AI Act Article 25]

Article 25 establishes that where a high-risk AI system is placed on the market or put into service, the provider is responsible for ensuring compliance with the obligations in this Regulation. Where a deployer uses a high-risk AI system, the deployer bears responsibility for ensuring that the system is used in accordance with the instructions. The responsible party for any AI-driven decision must be clearly identifiable. Contracts between providers and deployers shall clearly delineate responsibilities including obligations for monitoring, reporting and corrective action.

[/CHUNK]

[CHUNK]
[REGULATION: EU AI Act]
[ARTICLE: Article 50 — Transparency Obligations for Providers and Deployers of Certain AI Systems]
[REFERENCE: EU AI Act Article 50]

Article 50 requires that providers of AI systems intended to interact directly with natural persons shall ensure that those AI systems are designed and developed in such a way that the natural persons concerned are informed that they are interacting with an AI system. This obligation does not apply where the use of an AI system is authorised by law for law enforcement purposes. Deployers of AI systems that generate or manipulate image, audio or video content shall disclose that the content has been artificially generated or manipulated. AI-generated content in automated decision making that significantly affects individuals must be disclosed as such.

[/CHUNK]

[CHUNK]
[REGULATION: EU AI Act]
[ARTICLE: Article 8 — Compliance with Requirements — High-Risk AI Systems]
[REFERENCE: EU AI Act Article 8]

Article 8 establishes that high-risk AI systems shall comply with the requirements in Chapter III Section 2. In demonstrating compliance, providers shall take into account the intended purpose of the high-risk AI system as well as the generally acknowledged state of the art. High-risk AI systems include AI used in employment decisions, credit scoring, educational assessment, access to essential private services, and law enforcement. For each such system, providers must implement the full set of risk management, data governance, transparency, human oversight, and robustness requirements.

[/CHUNK]
```

---

### `rag/documents/gdpr.txt`

```
[CHUNK]
[REGULATION: GDPR]
[ARTICLE: Article 5 — Principles Relating to Processing of Personal Data]
[REFERENCE: GDPR Article 5]

Article 5 establishes that personal data shall be: processed lawfully, fairly and in a transparent manner in relation to the data subject (lawfulness, fairness and transparency); collected for specified, explicit and legitimate purposes and not further processed in a manner that is incompatible with those purposes (purpose limitation); adequate, relevant and limited to what is necessary in relation to the purposes for which they are processed (data minimisation); accurate and kept up to date (accuracy); kept in a form which permits identification of data subjects for no longer than is necessary (storage limitation); processed in a manner that ensures appropriate security of the personal data (integrity and confidentiality).

[/CHUNK]

[CHUNK]
[REGULATION: GDPR]
[ARTICLE: Article 13 — Information to be Provided where Personal Data are Collected from the Data Subject]
[REFERENCE: GDPR Article 13]

Article 13 requires that where personal data relating to a data subject are collected, the controller shall at the time when personal data are obtained provide the data subject with information including: the identity and contact details of the controller; the purposes of the processing and the legal basis; the recipients of the personal data; the period for which personal data will be stored. Where the controller intends to process personal data for a purpose other than that for which they were collected, the controller shall provide the data subject with information on that other purpose. AI systems using personal data for automated decision-making must provide this disclosure at the point of data collection.

[/CHUNK]

[CHUNK]
[REGULATION: GDPR]
[ARTICLE: Article 22 — Automated Individual Decision-Making, Including Profiling]
[REFERENCE: GDPR Article 22]

Article 22 establishes that the data subject shall have the right not to be subject to a decision based solely on automated processing, including profiling, which produces legal effects concerning him or her or similarly significantly affects him or her. This right does not apply if the decision is necessary for entering into or performance of a contract, is authorised by law, or is based on the data subject's explicit consent. Where automated decision-making is used, the controller shall implement suitable measures to safeguard the data subject's rights and freedoms and legitimate interests, including the right to obtain human intervention, to express his or her point of view and to contest the decision. AI systems making automated credit, employment or access decisions must clearly disclose the automated nature and provide meaningful information about the logic involved.

[/CHUNK]

[CHUNK]
[REGULATION: GDPR]
[ARTICLE: Article 9 — Processing of Special Categories of Personal Data]
[REFERENCE: GDPR Article 9]

Article 9 prohibits the processing of personal data revealing racial or ethnic origin, political opinions, religious or philosophical beliefs, trade union membership, genetic data, biometric data for uniquely identifying a natural person, data concerning health or data concerning a natural person's sex life or sexual orientation unless specific conditions are met. AI systems must not use, infer, or expose special category data in their outputs without explicit legal basis. Outputs that reveal or can be used to infer protected characteristics constitute a processing of special category data even if the model was not trained on such data explicitly.

[/CHUNK]
```

---

### `rag/documents/nist_ai_rmf.txt`

```
[CHUNK]
[REGULATION: NIST AI RMF]
[ARTICLE: GOVERN 1.1 — Policies and Processes for AI Risk]
[REFERENCE: NIST AI RMF GOVERN 1.1]

GOVERN 1.1: Policies, processes, procedures, and practices across the organisation related to the mapping, measuring, and managing of AI risks are in place, transparent, and implemented effectively. Organisational policies for responsible AI should document who is accountable for AI decisions, what approval process exists for deploying AI systems, how violations are escalated, and what the review cycle for AI policies is. AI-generated outputs in regulated domains must be traceable to a responsible organisational party.

[/CHUNK]

[CHUNK]
[REGULATION: NIST AI RMF]
[ARTICLE: GOVERN 6.1 — Policies for Accountability]
[REFERENCE: NIST AI RMF GOVERN 6.1]

GOVERN 6.1: Policies and procedures are in place to assess and manage risk in AI systems from third-party providers and within the supply chain. Accountability structures for AI decisions must clearly identify the responsible party. For each AI-generated output that significantly affects individuals, the organisation deploying the AI system must be identifiable, named, and reachable. Accountability gaps occur when AI outputs reference no responsible party, making corrective action or appeal impossible.

[/CHUNK]

[CHUNK]
[REGULATION: NIST AI RMF]
[ARTICLE: MAP 3.5 — Data Provenance and Limitations]
[REFERENCE: NIST AI RMF MAP 3.5]

MAP 3.5: Practices and personnel for supporting AI risk identification processes are in place. Data provenance and lineage are documented. AI systems must identify and disclose the sources of data used to train and validate models, acknowledge known limitations in training data, and communicate how those limitations may affect the reliability of outputs. Outputs from AI systems that make decisions based on incomplete, biased or temporally limited datasets must acknowledge these limitations to users and affected individuals.

[/CHUNK]

[CHUNK]
[REGULATION: NIST AI RMF]
[ARTICLE: MEASURE 2.2 — Non-Discrimination and Fairness Metrics]
[REFERENCE: NIST AI RMF MEASURE 2.2]

MEASURE 2.2: Scientific and statistical methods are applied for AI risk measurement. Non-discrimination in AI outputs requires measuring and reporting group fairness metrics across protected characteristics. Disparate impact analysis, demographic parity, and equalized odds are standard quantitative measures. Outputs from AI systems used in consequential decisions must not demonstrate statistically significant disparate impact across groups defined by race, gender, age, religion, disability or other protected characteristics. The 4/5ths (80%) rule for adverse impact is the established legal threshold in employment and credit contexts.

[/CHUNK]

[CHUNK]
[REGULATION: NIST AI RMF]
[ARTICLE: MEASURE 2.5 — Explainability]
[REFERENCE: NIST AI RMF MEASURE 2.5]

MEASURE 2.5: The risk or impact of the AI system to individuals, groups, communities, organisations, and society is evaluated and documented. AI systems making consequential decisions must provide explanations that are meaningful and accessible to affected individuals. Explanations should identify which input features most significantly influenced the outcome. For credit, employment, or criminal justice decisions, individuals must receive an explanation of the specific factors and their relative importance. Technical explainability methods include SHAP (SHapley Additive exPlanations), LIME, and integrated gradients.

[/CHUNK]

[CHUNK]
[REGULATION: NIST AI RMF]
[ARTICLE: MEASURE 2.6 — Uncertainty and Confidence]
[REFERENCE: NIST AI RMF MEASURE 2.6]

MEASURE 2.6: The risk or impact of the AI system to individuals is evaluated considering epistemic and aleatoric uncertainty. AI systems must communicate appropriate uncertainty or confidence levels with their outputs. Where a model produces a prediction with low confidence or high uncertainty, this must be disclosed to the user or affected individual. Confidence scores, probability estimates, or qualitative uncertainty indicators must accompany AI-generated decisions in high-stakes domains. Presenting AI outputs as certain when the model has high uncertainty constitutes a transparency failure.

[/CHUNK]

[CHUNK]
[REGULATION: NIST AI RMF]
[ARTICLE: MANAGE 1.3 — Human Oversight and Review]
[REFERENCE: NIST AI RMF MANAGE 1.3]

MANAGE 1.3: Responses to identified risks are prioritised and implemented based on projected impact. Human oversight mechanisms are a primary risk management control for high-risk AI systems. AI systems in consequential domains must provide a mechanism for human review, appeal, or override of automated decisions. The absence of a human review pathway is a significant governance gap. AI outputs that affect individuals' access to credit, employment, education, healthcare, or justice must inform affected individuals of their right to request human intervention.

[/CHUNK]
```

---

### `rag/documents/dpdpa_2023.txt`

```
[CHUNK]
[REGULATION: India DPDPA 2023]
[ARTICLE: Section 4 — Grounds for Processing Personal Data]
[REFERENCE: India DPDPA 2023 Section 4]

Section 4 of the Digital Personal Data Protection Act 2023 establishes that a Data Fiduciary may process the personal data of a Data Principal only in accordance with the provisions of this Act. Personal data may be processed only for a lawful purpose for which the Data Principal has given consent, or for certain legitimate uses specified under the Act. A Data Fiduciary must provide clear notice to the Data Principal about the personal data to be collected, the purpose of processing, and the manner in which consent may be withdrawn. AI systems that process personal data of Indian residents must establish a lawful purpose and obtain consent before processing.

[/CHUNK]

[CHUNK]
[REGULATION: India DPDPA 2023]
[ARTICLE: Section 6 — Notice and Consent]
[REFERENCE: India DPDPA 2023 Section 6]

Section 6 requires that every request for consent for processing personal data shall be accompanied or preceded by a notice by the Data Fiduciary to the Data Principal. The notice shall contain: a description of the personal data sought to be collected; the purpose of processing such personal data; the manner in which the Data Principal may exercise their rights; the manner in which the Data Principal may make a complaint. Consent must be free, specific, informed, unconditional and unambiguous with a clear affirmative action. AI systems must not process personal data beyond the scope of collected consent. Automated decisions based on personal data must be disclosed as such.

[/CHUNK]

[CHUNK]
[REGULATION: India DPDPA 2023]
[ARTICLE: Section 8 — Obligations of Data Fiduciary]
[REFERENCE: India DPDPA 2023 Section 8]

Section 8(3) requires that a Data Fiduciary shall ensure that the personal data processed is complete, accurate and consistent, to the extent necessary for the specified purpose. The Data Fiduciary shall not retain personal data beyond the period necessary for the specified purpose. Section 8(7) prohibits processing personal data that is not necessary for the specified purpose — this is the data minimisation obligation. AI systems must not collect or process personal data in excess of what is necessary for the stated decision-making purpose. Excess data collection or retention beyond the purpose constitutes a violation of data minimisation obligations under DPDPA.

[/CHUNK]

[CHUNK]
[REGULATION: India DPDPA 2023]
[ARTICLE: Section 11 — Right to Information about Personal Data]
[REFERENCE: India DPDPA 2023 Section 11]

Section 11 provides that a Data Principal shall have the right to obtain from the Data Fiduciary: confirmation of whether personal data is being processed; a summary of the personal data being processed; the identities of other Data Fiduciaries and processors with whom personal data has been shared. Where an AI system makes an automated decision affecting an Indian data principal, the data principal has the right to request information about what personal data was used in that decision and how it was processed. AI systems making consequential automated decisions must maintain records sufficient to respond to such information requests.

[/CHUNK]
```

---

### `rag/documents/rbi_guidelines.txt`

```
[CHUNK]
[REGULATION: RBI Model Risk Management Guidelines]
[ARTICLE: Section 3.2 — Model Validation]
[REFERENCE: RBI Model Risk Management Guidelines 2023 Section 3.2]

Section 3.2 of the Reserve Bank of India's Model Risk Management Guidelines requires that all models used in credit risk, market risk, and operational risk decisions shall be subject to independent validation before deployment and periodically thereafter. Model validation must assess: conceptual soundness of the modelling approach; ongoing monitoring of model performance; outcome analysis comparing predicted versus actual outcomes. For AI and machine learning models, validation must include assessment of stability across demographic groups, sensitivity to input perturbations, and performance against a benchmark or champion model. Models used in credit scoring, loan decisioning, and risk assessment must document their validation methodology in the model inventory.

[/CHUNK]

[CHUNK]
[REGULATION: RBI Model Risk Management Guidelines]
[ARTICLE: Section 4.1 — Model Governance Framework]
[REFERENCE: RBI Model Risk Management Guidelines 2023 Section 4.1]

Section 4.1 requires regulated entities to establish a comprehensive model governance framework. The framework must include: a model inventory documenting all models in production with risk ratings; model ownership and accountability structures; approval processes for model development, deployment and significant changes; ongoing performance monitoring with defined thresholds for recalibration or decommissioning. AI and ML models used in credit and financial decisions must undergo champion-challenger testing before full deployment. Any material deviation in model performance from established benchmarks must be escalated to senior management. Models that have not been validated, monitored against benchmarks, or documented in a model inventory do not meet RBI governance requirements.

[/CHUNK]

[CHUNK]
[REGULATION: RBI Model Risk Management Guidelines]
[ARTICLE: Section 5 — Data Quality and Governance for Models]
[REFERENCE: RBI Model Risk Management Guidelines 2023 Section 5]

Section 5 requires that the data used in model development, validation and ongoing monitoring must be subject to documented data governance controls. Data used to train AI and ML models in financial institutions must: have documented lineage including source systems and transformation steps; be assessed for completeness, accuracy and consistency; be evaluated for potential biases that may cause the model to produce discriminatory outcomes. Regulated entities must maintain records of the data used to train each model version, including the time period covered, exclusion criteria, and known limitations. AI models in credit scoring that use data with known demographic biases must document this limitation and demonstrate that appropriate bias mitigation steps have been taken.

[/CHUNK]
```

---

## Step 2 — Create `rag/__init__.py`

Empty file to make `rag/` a Python package:

```python
# rag/__init__.py
```

---

## Step 3 — Create `rag/retriever.py`

This module is imported by `policy_agent.py` at runtime. It handles store initialisation, query embedding, and similarity search. Designed to fail gracefully — never crashes the policy agent.

```python
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
from typing import Optional

logger = logging.getLogger(__name__)

_STORE_PATH = Path(__file__).parent / "policy_store"
_COLLECTION_NAME = "policy_regulations"
_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

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

    if not _STORE_PATH.exists():
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

        for doc, meta, dist in zip(documents, metadatas, distances):
            # ChromaDB returns L2 distance for cosine space; convert to similarity
            # With normalised embeddings, cosine similarity = 1 - (dist / 2)
            similarity = round(max(0.0, 1.0 - dist / 2.0), 4)

            clauses.append({
                "text": doc,
                "regulation": meta.get("regulation", "Unknown"),
                "article_id": meta.get("article_id", "Unknown"),
                "reference": meta.get("reference", ""),
                "similarity_score": similarity,
            })

        return clauses

    except Exception as e:
        logger.warning("[RAG] Retrieval failed: %s. Returning empty.", e)
        return []


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
```

---

## Step 4 — Create `rag/indexer.py`

This script is run once (or whenever documents change) to build the vector store. It must be runnable standalone: `python rag/indexer.py`.

```python
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

    # Show sample retrieval
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
```

---

## Step 5 — Modify `state.py`

Add `retrieved_clauses` to `ComplianceState`. This stores the regulatory chunks retrieved at each policy check in the audit trail.

**Locate the `# --- Agent results ---` block and add one line:**

```python
# --- Agent results (None until that agent runs) ---
pii_result: Optional[PIIResult]
bias_result: Optional[BiasResult]
policy_result: Optional[PolicyResult]
explainability_result: Optional[ExplainabilityResult]
retrieved_clauses: Optional[list[dict]]      # ADD THIS LINE
```

The field goes after `explainability_result`. It stores a list of dicts, each containing `{text, regulation, article_id, reference, similarity_score}`.

**In `create_initial_state()`, add the default value:**

```python
# Agent results (all None until agents run)
pii_result=None,
bias_result=None,
policy_result=None,
explainability_result=None,
retrieved_clauses=None,          # ADD THIS LINE
```

Add it after `explainability_result=None,`.

---

## Step 6 — Modify `nodes/policy_agent.py`

Four targeted changes. Do not restructure the existing code — make surgical additions only.

### 6.1 — Add import at the top of `_check_policy_compliance()`

At the start of `_check_policy_compliance()`, before the `ChatOllama` instantiation, add the retrieval call:

```python
def _check_policy_compliance(state: ComplianceState) -> tuple[PolicyResult, list[dict]]:
    """
    Calls Ollama/Gemma to evaluate current_text against POLICY_CRITERIA.
    Also retrieves relevant regulatory clauses from ChromaDB vector store.
    Returns (PolicyResult, retrieved_clauses).
    """
    from langchain_ollama import ChatOllama
    from rag.retriever import retrieve_relevant_clauses   # ADD THIS IMPORT

    llm = ChatOllama(model=OLLAMA_MODEL, temperature=0)

    # --- Retrieve relevant regulatory clauses (RAG) ---
    retrieved = retrieve_relevant_clauses(state["current_text"], k=5)
    if retrieved:
        print(f"[POLICY AGENT] Retrieved {len(retrieved)} regulatory clauses from vector store.")
        for c in retrieved[:3]:
            print(f"               → {c['regulation']} {c['article_id']} (sim={c['similarity_score']})")
    else:
        print("[POLICY AGENT] RAG store not available — using hardcoded criteria only.")
    # --- End RAG retrieval ---

    try:
        structured_llm = llm.with_structured_output(PolicyCheckOutput)
        prompt = build_policy_prompt(state["current_text"], POLICY_CRITERIA, retrieved)   # pass retrieved
        ...
```

### 6.2 — Update `policy_agent_node()` to handle the new return signature

Change the call to `_check_policy_compliance` and store `retrieved_clauses` in the returned state:

```python
def policy_agent_node(state: ComplianceState) -> dict:
    print(f"\n[POLICY AGENT] Checking regulatory compliance...")

    policy_result, retrieved_clauses = _check_policy_compliance(state)   # unpack tuple

    new_violations = (
        [f"POLICY_{v['id']}" for v in policy_result["violations"]]
        if policy_result["violations"] else []
    )

    status = "FAIL" if not policy_result["passed"] else "PASS"
    print(f"[POLICY AGENT] Result  : {status}")
    print(f"[POLICY AGENT] Severity: {policy_result['severity']}")
    for v in policy_result["violations"]:
        print(f"               → [{v['severity'].upper()}] {v['id']}: {v['description']}")

    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "node": "policy_agent",
        "action": "POLICY_CHECK",
        "result": "fail" if not policy_result["passed"] else "pass",
        "detail": {
            "violations_found": len(policy_result["violations"]),
            "severity": policy_result["severity"],
            "violation_ids": [v["id"] for v in policy_result["violations"]],
            "rag_clauses_retrieved": len(retrieved_clauses),          # ADD
            "rag_sources": list({c["regulation"] for c in retrieved_clauses}),  # ADD
        },
        "correction_count": state["correction_count"],
    }

    return {
        "policy_result": policy_result,
        "retrieved_clauses": retrieved_clauses,           # ADD
        "violations": new_violations,
        "current_node": "policy_agent",
        "audit_log": [log_entry],
    }
```

### 6.3 — Update `_check_policy_compliance()` return type and fallback

```python
def _check_policy_compliance(state: ComplianceState) -> tuple[PolicyResult, list[dict]]:
    from langchain_ollama import ChatOllama
    from rag.retriever import retrieve_relevant_clauses

    llm = ChatOllama(model=OLLAMA_MODEL, temperature=0)
    retrieved = retrieve_relevant_clauses(state["current_text"], k=5)

    if retrieved:
        print(f"[POLICY AGENT] Retrieved {len(retrieved)} regulatory clauses from vector store.")
    else:
        print("[POLICY AGENT] RAG store not available — using hardcoded criteria only.")

    try:
        structured_llm = llm.with_structured_output(PolicyCheckOutput)
        prompt = build_policy_prompt(state["current_text"], POLICY_CRITERIA, retrieved)
        result: PolicyCheckOutput = structured_llm.invoke(prompt)

        violations = [
            {
                "id": v.id,
                "description": v.description,
                "severity": v.severity,
                "article_reference": v.article_reference,
                "remediation": v.remediation,
            }
            for v in result.violations
            if v.id in VALID_IDS
        ]

        passed = len(violations) == 0
        severity = (
            max((v["severity"] for v in violations), key=lambda s: SEVERITY_ORDER[s])
            if violations else "none"
        )

        return PolicyResult(
            violations=violations,
            passed=passed,
            severity=severity,
            summary=result.summary,
        ), retrieved

    except Exception as e:
        print(f"[POLICY AGENT] LLM call failed: {e}. Returning safe fallback.")
        return _safe_fallback(str(e)), retrieved   # still return retrieved even on LLM error
```

Also update `_safe_fallback` call in the except block to match.

### 6.4 — Update `build_policy_prompt()` to accept and format retrieved clauses

Replace the existing `build_policy_prompt` function entirely:

```python
def build_policy_prompt(text: str, criteria: list[dict], retrieved_clauses: list[dict] = None) -> str:
    """
    Builds the policy compliance prompt.
    If retrieved_clauses is provided, appends them as regulatory context
    to ground the LLM's analysis in actual article language.
    """
    criteria_text = "\n".join(
        f"  - {c['id']}: {c['description']} (Ref: {c['reference']})"
        for c in criteria
    )

    rag_section = ""
    if retrieved_clauses:
        rag_lines = []
        for i, clause in enumerate(retrieved_clauses, 1):
            rag_lines.append(
                f"  [{i}] {clause['regulation']} — {clause['article_id']} "
                f"(relevance: {clause['similarity_score']:.2f})\n"
                f"      {clause['text'][:400]}..."
            )
        rag_section = (
            "\n\nRETRIEVED REGULATORY CONTEXT "
            "(most relevant clauses retrieved from regulatory documents):\n"
            + "\n".join(rag_lines)
            + "\n\nUse the retrieved regulatory context above to ground your analysis. "
            "When identifying a violation, you may reference specific language from "
            "the retrieved clauses. However, only flag criteria from the REGULATORY "
            "CRITERIA list — do not invent new criteria from the retrieved context."
        )

    return f"""You are a Responsible AI compliance auditor evaluating AI-generated text.

TEXT TO EVALUATE:
\"\"\"{text}\"\"\"

REGULATORY CRITERIA:
{criteria_text}
{rag_section}

IMPORTANT RULES:
- Flag a criterion ONLY if it is CLEARLY AND EXPLICITLY violated.
- Do NOT flag PRIVACY unless the text contains real personal identifiers (names, emails, phone numbers, ID numbers). Aggregate statistics and anonymised metrics are NOT privacy violations.
- Do NOT flag ACCOUNTABILITY if a responsible organisation is named, even without a specific contact number. A named institution satisfies this criterion.
- Do NOT flag DATA_QUALITY if the text names at least one data source and mentions at least one limitation.
- Do NOT speculate about what the text could imply. Only flag what it explicitly does or omits.
- If in doubt, do NOT flag. Return an empty violations array for borderline cases.
- Do NOT flag DPDPA_CONSENT unless the text explicitly processes Indian user personal data without mentioning consent, lawful purpose, or disclosure.
- Do NOT flag DPDPA_DATA_MINIMISATION unless the text explicitly references collecting more data than the stated purpose requires.
- Do NOT flag RBI_MODEL_VALIDATION unless the text is clearly about a financial model deployment and contains no mention of validation, monitoring, or benchmarking methodology.

For each clear violation return:
  - id: criterion ID exactly as listed in REGULATORY CRITERIA above
  - description: one sentence explaining what is violated, citing retrieved regulatory language where relevant
  - severity: "low", "medium", or "high"
  - article_reference: the reference string from the criterion
  - remediation: one sentence concrete fix

Return a JSON object with:
  - violations: array of violation objects (empty array [] if none)
  - summary: one paragraph plain-English compliance summary, referencing specific retrieved articles where relevant

Return ONLY valid JSON. No markdown fences, no text outside the JSON."""
```

---

## Step 7 — Modify `ui/app.py`

Add a "Regulatory Sources" expander in the Text Auditor results section. This is the most visible part of the RAG demo — it shows the panel that the system is grounding its analysis in actual retrieved regulation text.

**In the `render_text_auditor()` function, after the policy violations expander, add:**

```python
# Show retrieved regulatory clauses (RAG evidence)
retrieved = final_state.get("retrieved_clauses") or []
if retrieved:
    with st.expander(f"Regulatory Sources — {len(retrieved)} clauses retrieved", expanded=False):
        st.caption(
            "These regulatory clauses were retrieved from the vector store based on semantic "
            "relevance to the audited text. They grounded the policy compliance analysis."
        )
        for i, clause in enumerate(retrieved, 1):
            sim_pct = int(clause.get("similarity_score", 0) * 100)
            colour = "#28a745" if sim_pct >= 60 else "#ffc107" if sim_pct >= 40 else "#6c757d"
            st.markdown(
                f'<div style="border-left:3px solid {colour};padding:8px 12px;margin-bottom:8px;">'
                f'<strong>{clause["regulation"]} — {clause["article_id"]}</strong> '
                f'<span style="color:{colour};font-size:0.8rem;">({sim_pct}% relevance)</span><br>'
                f'<span style="font-size:0.85rem;color:#555;">{clause["reference"]}</span><br><br>'
                f'<span style="font-size:0.85rem;">{clause["text"][:350]}...</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
```

Add the same block to `render_model_auditor()` for the model audit tab, using the same `retrieved_clauses` from `final`.

---

## Step 8 — Create `tests/test_rag_retriever.py`

```python
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
# Store state tests
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
```

---

## Step 9 — Update `CLAUDE.md` Commands Section

Add these lines to the Commands section of `CLAUDE.md`:

```bash
# Build RAG vector store (run once after creating rag/documents/)
python rag/indexer.py

# Rebuild from scratch (e.g. after updating document files)
python rag/indexer.py --reset

# Verify store and inspect sample retrievals
python rag/indexer.py --verify

# RAG retriever tests (requires store to be built first)
python -m pytest tests/test_rag_retriever.py -v
```

---

## Completion Checklist

Run in this exact order:

```bash
# 1. Create directory structure
mkdir -p rag/documents rag/policy_store

# 2. Create the 5 document files (eu_ai_act.txt, gdpr.txt, nist_ai_rmf.txt,
#    dpdpa_2023.txt, rbi_guidelines.txt) using the content in Steps 1a-1e above

# 3. Create rag/__init__.py, rag/retriever.py, rag/indexer.py

# 4. Build the vector store
python rag/indexer.py
# Expected output: ~20 chunks indexed across 5 regulation files

# 5. Verify store and sample retrievals
python rag/indexer.py --verify
# Expected: 3 relevant results for each sample query

# 6. Update state.py — add retrieved_clauses field
# 7. Update nodes/policy_agent.py — all 4 changes in Step 6
# 8. Update ui/app.py — add Regulatory Sources expander
# 9. Create tests/test_rag_retriever.py

# 10. Run retriever tests
python -m pytest tests/test_rag_retriever.py -v
# All store-ready tests should pass

# 11. Run full demo to verify end-to-end behaviour
python main.py
# Watch for: "[POLICY AGENT] Retrieved 5 regulatory clauses from vector store."
# And:       "rag_clauses_retrieved": 5 in the audit log

# 12. Run Scenario 1 specifically to confirm no regressions
# Expected: PASS — retrieved clauses should not cause false positives

# 13. Launch UI and verify Regulatory Sources expander appears
streamlit run ui/app.py
# In Text Auditor tab: paste Scenario 1 text, run audit
# Expand "Regulatory Sources" — should show 5 clause cards with colour-coded relevance bars

# 14. Run full test suite
python -m pytest tests/ -v --ignore=tests/test_policy_agent.py --ignore=tests/test_integration.py
# (policy and integration tests require Ollama running)
```

---

## Design Constraints to Preserve

- The fallback in `retrieve_relevant_clauses()` that returns `[]` when the store is missing must stay. If someone runs the demo without building the store, the policy agent must still work — it just uses hardcoded criteria only.
- The `VALID_IDS` filter in `_check_policy_compliance()` must remain. Retrieved clause context might tempt the LLM to generate violation IDs not in the criteria list. The filter blocks this.
- The `_safe_fallback` path must still return `(PolicyResult, [])` as a tuple after this change. Do not let the LLM error path crash the graph.
- Do not use ChromaDB's default in-memory client anywhere in production code paths — always use `PersistentClient`. In-memory stores are lost between restarts and will cause confusion during the demo.
- `retrieved_clauses` in state uses direct assignment (not `operator.add`). Only the policy agent writes this field. It does not accumulate across cycles — each policy check overwrites the previous run's retrieved clauses.

---

## File Change Summary

| File | Action |
|---|---|
| `rag/__init__.py` | Create (empty package marker) |
| `rag/retriever.py` | Create — ChromaDB query wrapper with graceful fallback |
| `rag/indexer.py` | Create — document parsing and indexing script |
| `rag/documents/eu_ai_act.txt` | Create — EU AI Act articles 5, 8, 9, 10, 13, 14, 15, 25, 50 |
| `rag/documents/gdpr.txt` | Create — GDPR articles 5, 9, 13, 22 |
| `rag/documents/nist_ai_rmf.txt` | Create — NIST GOVERN 1.1, 6.1, MAP 3.5, MEASURE 2.2, 2.5, 2.6, MANAGE 1.3 |
| `rag/documents/dpdpa_2023.txt` | Create — DPDPA Sections 4, 6, 8, 11 |
| `rag/documents/rbi_guidelines.txt` | Create — RBI Guidelines Sections 3.2, 4.1, 5 |
| `state.py` | Add `retrieved_clauses: Optional[list[dict]]` field + default in factory |
| `nodes/policy_agent.py` | Import retriever; update `_check_policy_compliance` return type; update `policy_agent_node` to store retrieved clauses; update `build_policy_prompt` to accept and format clauses |
| `ui/app.py` | Add Regulatory Sources expander in both tabs |
| `tests/test_rag_retriever.py` | Create — 10 tests with store-ready guards |
| `CLAUDE.md` | Add RAG commands to Commands section |
