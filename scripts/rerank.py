"""LLM re-ranker: given an invoice line + top-k BM25 candidates (already size-filtered),
pick the matching catalog product or say none match, with a confidence the gate can act on.

Stdlib only. Reads ANTHROPIC_API_KEY from the environment. Responses cached in data/llm_cache.jsonl.
"""
import json, os, hashlib, urllib.request

MODEL = os.environ.get("RERANK_MODEL", "claude-haiku-5-5")
CACHE = "data/llm_cache.jsonl"

SYSTEM = """You match lines from liquor distributor invoices to a store's product catalog.
Size and pack already match for every candidate. Decide whether the invoice line is the SAME
product as one candidate. Ignore catalog noise such as status prefixes (e.g. SOOH, CM, HA),
DISCO, PET (plastic bottle), and "USE CODE n" notes. Flavors, ages (4YR vs 12YR), proof, gift
sets and different expressions are DIFFERENT products. If none is the same product, answer 0."""

TOOL = {
    "name": "pick_match",
    "description": "Return the matching candidate number, or 0 if none match.",
    "input_schema": {
        "type": "object",
        "properties": {
            "choice": {"type": "integer", "description": "1-based candidate number, or 0 for no match"},
            "confidence": {"type": "number", "description": "0.0-1.0 probability the choice is correct"},
            "reason": {"type": "string", "description": "One short sentence"},
        },
        "required": ["choice", "confidence", "reason"],
    },
}

def prompt(line, cands):
    rows = "\n".join(f"{i}. {name}" for i, (_, name, _) in enumerate(cands, 1))
    return f"Invoice line: {line}\n\nCandidates:\n{rows}"

_cache = None
def _load():
    global _cache
    if _cache is None:
        _cache = {}
        if os.path.exists(CACHE):
            for l in open(CACHE):
                d = json.loads(l); _cache[d["h"]] = d["out"]
    return _cache

def call(line, cands, dry=False):
    p = prompt(line, cands)
    h = hashlib.sha256(f"{MODEL}\n{SYSTEM}\n{p}".encode()).hexdigest()
    cache = _load()
    if h in cache: return cache[h]
    if dry: return {"choice": 1, "confidence": 0.0, "reason": "dry run", "_prompt": p}
    body = {
        "model": MODEL, "max_tokens": 200, "system": SYSTEM,
        "tools": [TOOL], "tool_choice": {"type": "tool", "name": "pick_match"},
        "messages": [{"role": "user", "content": p}],
    }
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages", data=json.dumps(body).encode(),
        headers={"x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01",
                 "content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        resp = json.load(r)
    out = next(b["input"] for b in resp["content"] if b["type"] == "tool_use")
    cache[h] = out
    with open(CACHE, "a") as f: f.write(json.dumps({"h": h, "out": out}) + "\n")
    return out
