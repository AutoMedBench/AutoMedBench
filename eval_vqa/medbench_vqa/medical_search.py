"""Bounded medical literature search helper for VQA agents.

Queries NCBI E-utilities (PubMed / PMC / MedlinePlus) through a strict URL
whitelist.  Leak-prevention is done via a blocklist of benchmark-related
phrases in the query.  Designed to be safe to invoke from agent-generated
Python code inside the sandbox.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from time import perf_counter
from typing import Any

from ._logger import record_tool_call

ALLOWED_DOMAINS = {
    "eutils.ncbi.nlm.nih.gov",
    "www.ncbi.nlm.nih.gov",
    "medlineplus.gov",
}
BLOCKED_PHRASES = {
    "answer",
    "answer key",
    "benchmark",
    "correct answer",
    "ground truth",
    "huggingface",
    "gold answer",
    "kaggle",
    "medframeqa",
    "pathvqa",
    "reference answer",
    "vqa-rad",
    "vqarad",
    "medxpertqa",
}

_WHITESPACE_RE = re.compile(r"\s+")


def _sanitize_query(query: str) -> str:
    q = _WHITESPACE_RE.sub(" ", (query or "").strip()).lower()
    for phrase in sorted(BLOCKED_PHRASES, key=len, reverse=True):
        q = q.replace(phrase, "")
    return _WHITESPACE_RE.sub(" ", q).strip()


def public_medical_search(
    query: str,
    *,
    max_results: int = 5,
    db: str = "pubmed",
    timeout_s: float = 20.0,
) -> dict[str, Any]:
    """Search PubMed (default) or other NCBI databases and return summaries.

    Returns ``{"status", "query", "sanitized_query", "results": [...]}``.
    Each result entry has ``pmid``, ``title``, ``authors``, ``journal``,
    ``year``, ``url``.  The helper logs the call into tool_calls.jsonl.
    """
    started = perf_counter()
    sanitized = _sanitize_query(query)
    if not sanitized:
        result = {
            "status": "rejected",
            "reason": "query empty after sanitization",
            "query": query,
            "sanitized_query": sanitized,
            "results": [],
        }
        record_tool_call(
            tool="public_medical_search",
            arguments={"query": query, "max_results": max_results, "db": db},
            result_summary={"status": "rejected", "result_count": 0},
        )
        return result

    try:
        pmids = _esearch(sanitized, db=db, max_results=max_results, timeout=timeout_s)
        summaries = _esummary(pmids, db=db, timeout=timeout_s) if pmids else []
        results = [
            {
                "pmid": item.get("uid"),
                "title": item.get("title"),
                "authors": [a.get("name") for a in (item.get("authors") or [])][:5],
                "journal": item.get("fulljournalname") or item.get("source"),
                "year": (item.get("pubdate") or "").split(" ")[0],
                "url": f"https://www.ncbi.nlm.nih.gov/{db}/{item.get('uid')}/",
            }
            for item in summaries
        ]
        payload = {
            "status": "ok",
            "query": query,
            "sanitized_query": sanitized,
            "db": db,
            "latency_ms": round((perf_counter() - started) * 1000, 2),
            "results": results,
        }
    except urllib.error.URLError as exc:
        payload = {
            "status": "network_error",
            "query": query,
            "sanitized_query": sanitized,
            "db": db,
            "error": str(exc),
            "results": [],
        }
    except Exception as exc:  # noqa: BLE001
        payload = {
            "status": "error",
            "query": query,
            "sanitized_query": sanitized,
            "db": db,
            "error": str(exc),
            "results": [],
        }

    record_tool_call(
        tool="public_medical_search",
        arguments={"query": query, "max_results": max_results, "db": db},
        result_summary={
            "status": payload["status"],
            "result_count": len(payload["results"]),
        },
    )
    return payload


def _esearch(query: str, *, db: str, max_results: int, timeout: float) -> list[str]:
    params = {
        "db": db,
        "term": query,
        "retmax": str(max_results),
        "retmode": "json",
    }
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?" + urllib.parse.urlencode(params)
    _assert_allowed(url)
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return list(data.get("esearchresult", {}).get("idlist", []))


def _esummary(pmids: list[str], *, db: str, timeout: float) -> list[dict[str, Any]]:
    if not pmids:
        return []
    params = {
        "db": db,
        "id": ",".join(pmids),
        "retmode": "json",
    }
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?" + urllib.parse.urlencode(params)
    _assert_allowed(url)
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    summary = data.get("result", {})
    return [summary[uid] for uid in summary.get("uids", []) if uid in summary]


def _assert_allowed(url: str) -> None:
    host = urllib.parse.urlparse(url).hostname or ""
    if host not in ALLOWED_DOMAINS:
        raise RuntimeError(f"Host {host!r} not in medical-search whitelist {sorted(ALLOWED_DOMAINS)}")
