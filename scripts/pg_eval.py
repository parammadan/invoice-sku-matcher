"""Run the size-filtered embedding search inside Postgres + pgvector and check it matches Python.

Needs a Postgres 16+ with pgvector reachable via psql. Defaults to the project-local instance:
  PGHOST=/tmp PGPORT=5433 PGUSER=matcher PGDATABASE=postgres
"""
import csv, json, os, subprocess, sys, time
sys.path.insert(0, "scripts")
from matcher import BM25Matcher, index_name
from equiv import build

ENV = {**os.environ, "PGHOST": os.environ.get("PGHOST", "/tmp"), "PGPORT": os.environ.get("PGPORT", "5433"),
       "PGUSER": os.environ.get("PGUSER", "matcher"), "PGDATABASE": os.environ.get("PGDATABASE", "postgres")}
PSQL = os.environ.get("PSQL", "/opt/homebrew/opt/postgresql@16/bin/psql")

def psql(sql):
    return subprocess.run([PSQL, "-v", "ON_ERROR_STOP=1", "-qAt", "-F", "\t"], input=sql, env=ENV,
                          capture_output=True, text=True, check=True).stdout

vec = lambda v: "[" + ",".join(f"{x:.6f}" for x in v) + "]"
rows = list(csv.DictReader(open("data/catalog.csv")))
gold = [json.loads(l) for l in open("data/gold.jsonl")]
cat, qv = json.load(open("data/emb_catalog.json")), json.load(open("data/emb_queries.json"))

os.makedirs("data/pgload", exist_ok=True)
with open("data/pgload/products.tsv", "w") as f:
    for r in rows:
        f.write("\t".join([r["key"], index_name(r["name"]), r["pack"], r["volume_ml"], vec(cat[r["key"]])]) + "\n")
with open("data/pgload/queries.tsv", "w") as f:
    for i, g in enumerate(gold):
        f.write("\t".join([str(i), g["query"], g["pack"], g["volume_ml"], g["target"], vec(qv[g["query"]])]) + "\n")

t = time.time()
psql(f"""
CREATE EXTENSION IF NOT EXISTS vector;
DROP TABLE IF EXISTS products, queries;
CREATE TABLE products (key text PRIMARY KEY, name text, pack int, volume_ml int, embedding vector(768));
CREATE TABLE queries (id int PRIMARY KEY, query text, pack int, volume_ml int, target text, embedding vector(768));
\\copy products FROM '{os.path.abspath("data/pgload/products.tsv")}'
\\copy queries FROM '{os.path.abspath("data/pgload/queries.tsv")}'
CREATE INDEX ON products (pack, volume_ml);
CREATE INDEX ON products USING hnsw (embedding vector_cosine_ops);
ANALYZE products;
""")
print(f"loaded {len(rows)} products + {len(gold)} queries, built indexes in {time.time()-t:.1f}s")

TOPK = """
SELECT q.id, p.key FROM queries q CROSS JOIN LATERAL (
  SELECT key FROM products p WHERE p.pack = q.pack AND p.volume_ml = q.volume_ml
  ORDER BY p.embedding <=> q.embedding LIMIT 5) p;"""
# relaxed_order may return candidates slightly out of order: over-fetch, then re-sort exactly
TOPK_RESORT = """
SELECT q.id, p.key FROM queries q CROSS JOIN LATERAL (
  SELECT key FROM (
    SELECT key, p.embedding <=> q.embedding AS d FROM products p
    WHERE p.pack = q.pack AND p.volume_ml = q.volume_ml
    ORDER BY p.embedding <=> q.embedding LIMIT 20) c
  ORDER BY d LIMIT 5) p;"""
same, _ = build(BM25Matcher("data/catalog.csv").rows)

def score(out):
    top = {}
    for line in out.strip().splitlines():
        i, key = line.split("\t"); top.setdefault(int(i), []).append(key)
    n = len(gold)
    t1 = sum(same(top[i][0], g["target"]) for i, g in enumerate(gold) if i in top)
    t5 = sum(any(same(k, g["target"]) for k in top.get(i, [])) for i, g in enumerate(gold))
    return t1 / n, t5 / n, top

t = time.time(); exact = score(psql(TOPK)); te = time.time() - t
print(f"exact (size filter via btree, then sort by distance): top-1 {exact[0]:.1%}  top-5 {exact[1]:.1%}  {te*1000/len(gold):.1f} ms/query")

# Force the HNSW index: filtered ANN is where pgvector recall can silently drop.
ann_sql = "SET enable_seqscan=off; SET enable_bitmapscan=off; SET enable_indexscan=on; DROP INDEX IF EXISTS products_pack_volume_ml_idx;"
for mode, sql in (("off", TOPK), ("relaxed_order", TOPK_RESORT)):
    t = time.time()
    ann = score(psql(ann_sql + f"SET hnsw.iterative_scan={mode}; SET hnsw.ef_search=40;" + sql))
    ta = time.time() - t
    agree = sum(i in ann[2] and same(ann[2][i][0], exact[2][i][0]) for i in exact[2]) / len(gold)  # ties: identical-name dupes
    missing = sum(len(ann[2].get(i, [])) < len(exact[2][i]) for i in exact[2])
    print(f"HNSW, iterative_scan={mode:13}: top-1 {ann[0]:.1%}  top-5 {ann[1]:.1%}  same top-1 product as exact {agree:.1%}  "
          f"queries short of 5 results {missing}  {ta*1000/len(gold):.1f} ms/query")
psql("CREATE INDEX IF NOT EXISTS products_pack_volume_ml_idx ON products (pack, volume_ml);")
