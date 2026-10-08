"""Embed catalog names + gold queries with a local Ollama model (default nomic-embed-text).
nomic expects task prefixes: 'search_document: ' for catalog, 'search_query: ' for queries.
Catalog names are embedded without redirect notes (matcher.index_name). Vectors are L2-normalized
and cached by exact input text in data/emb_cache.json, so re-runs only embed what changed.
Writes data/emb_catalog.json {key: vec} and data/emb_queries.json {query: vec}."""
import csv, json, math, os, sys, urllib.request
sys.path.insert(0, "scripts")
from matcher import index_name

MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")
CACHE = f"data/emb_cache_{MODEL.replace(':', '_')}.json"

def _call(texts):
    req = urllib.request.Request("http://localhost:11434/api/embed",
        data=json.dumps({"model": MODEL, "input": texts}).encode(), headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return [[x / (math.sqrt(sum(y * y for y in v)) or 1) for x in v] for v in json.load(r)["embeddings"]]

def embed(texts, cache, batch=64):
    todo = sorted({t for t in texts if t not in cache})
    for i in range(0, len(todo), batch):
        cache.update(zip(todo[i:i + batch], _call(todo[i:i + batch])))
        print(f"\r{min(i + batch, len(todo))}/{len(todo)} new", end="", file=sys.stderr)
    if todo: print(file=sys.stderr)
    return [cache[t] for t in texts], len(todo)

if __name__ == "__main__":
    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    rows = list(csv.DictReader(open("data/catalog.csv")))
    qs = sorted({json.loads(l)["query"] for l in open("data/gold.jsonl")})
    cv, n1 = embed(["search_document: " + index_name(r["name"]) for r in rows], cache)
    qv, n2 = embed(["search_query: " + q for q in qs], cache)
    json.dump(cache, open(CACHE, "w"))
    json.dump({r["key"]: v for r, v in zip(rows, cv)}, open("data/emb_catalog.json", "w"))
    json.dump(dict(zip(qs, qv)), open("data/emb_queries.json", "w"))
    print(f"{len(rows)} catalog + {len(qs)} queries ({n1 + n2} newly embedded), dim {len(cv[0])}")
