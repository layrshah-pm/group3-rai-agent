"""
utils/audit_logger.py
---------------------
Helper functions to query the SQLite audit database (rai_audit.db).

The database is written by LangGraph's SqliteSaver checkpointer using
msgpack serialisation. These helpers decode checkpointed state and extract
audit_log entries for display in the Streamlit UI.
"""

import sqlite3
from datetime import datetime
from pathlib import Path

import msgpack

DB_PATH = Path(__file__).parent.parent / "rai_audit.db"


def _connect() -> sqlite3.Connection:
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"Audit DB not found at {DB_PATH}. Run main.py first."
        )
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _decode_checkpoint(raw: bytes) -> dict:
    """Decode a msgpack-serialised checkpoint blob."""
    return msgpack.unpackb(raw, raw=False)


def _channel_values(raw: bytes) -> dict:
    try:
        data = _decode_checkpoint(raw)
        return data.get("channel_values", {})
    except Exception:
        return {}


def get_audit_trail(thread_id: str) -> list[dict]:
    """Returns all audit log entries for a given run, in chronological order."""
    conn = _connect()
    try:
        cursor = conn.execute(
            "SELECT checkpoint FROM checkpoints WHERE thread_id = ? ORDER BY rowid",
            (thread_id,),
        )
        rows = cursor.fetchall()
    finally:
        conn.close()

    entries = []
    seen: set[tuple] = set()
    for row in rows:
        cv = _channel_values(row["checkpoint"])
        for entry in cv.get("audit_log", []):
            key = (entry.get("timestamp"), entry.get("node"))
            if key not in seen:
                seen.add(key)
                entries.append(entry)

    return sorted(entries, key=lambda e: e.get("timestamp", ""))


def get_runs_summary() -> list[dict]:
    """Returns a summary of all past runs: thread_id, status, timestamp, total score."""
    conn = _connect()
    try:
        cursor = conn.execute(
            """
            SELECT thread_id, MAX(rowid) as last_rowid, checkpoint
            FROM checkpoints
            GROUP BY thread_id
            ORDER BY last_rowid DESC
            """
        )
        rows = cursor.fetchall()
    finally:
        conn.close()

    summaries = []
    for row in rows:
        cv = _channel_values(row["checkpoint"])
        final_status = cv.get("final_status", "pending")
        if final_status == "pending":
            continue
        audit_log = cv.get("audit_log", [])
        timestamp = audit_log[-1]["timestamp"] if audit_log else None
        rai_scores = cv.get("rai_scores") or {}
        summaries.append({
            "thread_id": row["thread_id"],
            "final_status": final_status,
            "timestamp": timestamp,
            "total_score": sum(rai_scores.values()) if rai_scores else None,
            "violations": list(set(cv.get("violations", []))),
            "correction_count": cv.get("correction_count", 0),
        })

    return summaries


def get_node_timings(thread_id: str) -> dict:
    """Returns approximate execution time per node based on audit log timestamps."""
    entries = get_audit_trail(thread_id)
    timings: dict[str, float] = {}

    for i, entry in enumerate(entries[:-1]):
        node = entry.get("node", "unknown")
        ts = entry.get("timestamp")
        next_ts = entries[i + 1].get("timestamp")
        if ts and next_ts:
            try:
                t0 = datetime.fromisoformat(ts)
                t1 = datetime.fromisoformat(next_ts)
                timings[node] = round((t1 - t0).total_seconds(), 3)
            except ValueError:
                pass

    return timings
