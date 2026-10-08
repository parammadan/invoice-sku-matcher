"""Reproduce every number in the README. Run from repo root after the data steps."""
import json, sys, subprocess
sys.path.insert(0, "scripts")
from matcher import BM25Matcher
from overlap_rerank import rerank
from equiv import build
from hybrid import Hybrid

gold = [json.loads(l) for l in open("data/gold.jsonl")]
m = BM25Matcher("data/catalog.csv")
same, _ = build(m.rows)
name_of = {r["key"]: r["name"] for r in m.rows}
n = len(gold)
print(f"catalog {len(m.rows)} SKUs | gold {n} real queries\n")

print("1) matching (redirect-aware scoring)")
h = Hybrid(m)
methods = {
    "BM25, text only": lambda g: [k for k, _, _ in m.match(g["query"], k=10)],
    "BM25 + size filter": lambda g: [k for k, _, _ in m.match(g["query"], g["pack"], g["volume_ml"], k=10)],
    "embeddings + size filter": lambda g: h.dense(g["query"], g["pack"], g["volume_ml"]),
    "hybrid RRF + size filter": lambda g: h.fused(g["query"], g["pack"], g["volume_ml"]),
}
tops = {}
for label, f in methods.items():
    out = [f(g) for g in gold]; tops[label] = [o[0] for o in out]
    top1 = sum(same(o[0], g["target"]) for o, g in zip(out, gold))
    hit5 = sum(any(same(k, g["target"]) for k in o[:5]) for o, g in zip(out, gold))
    wrong = sum((not same(o[0], g["target"])) and name_of[o[0]] == name_of[g["target"]] and o[0].split("|", 1)[1] != g["target"].split("|", 1)[1] for o, g in zip(out, gold))
    print(f"   {label:26} top-1 {top1/n:6.1%}   in top-5 {hit5/n:6.1%}   wrong-size top-1 {wrong/n:5.1%}")
b, e = tops["BM25 + size filter"], tops["embeddings + size filter"]
print(f"   only BM25 right {sum(same(x, g['target']) and not same(y, g['target']) for x, y, g in zip(b, e, gold))}"
      f" | only embeddings right {sum(same(y, g['target']) and not same(x, g['target']) for x, y, g in zip(b, e, gold))}")

print("\n2) confidence gate (overlap margin on size-filtered top-10)")
res = []
for g in gold:
    s, margin = rerank(m, g["query"], m.match(g["query"], g["pack"], g["volume_ml"], k=10))
    agree = same(s[0][1], h.dense(g["query"], g["pack"], g["volume_ml"], k=1)[0])
    res.append((same(s[0][1], g["target"]), margin, agree))
gates = [(f"margin >= {t}", lambda r, t=t: r[1] >= t) for t in (0.0, 0.1, 0.2, 0.3)]
gates += [("BM25 & embeddings agree", lambda r: r[2]), ("agree AND margin >= 0.3", lambda r: r[2] and r[1] >= 0.3)]
for label, f in gates:
    a = [r[0] for r in res if f(r)]
    print(f"   {label:24} auto-accept {len(a)/n:6.1%}   precision {sum(a)/len(a):6.1%}   to review {n-len(a)}")

print("\n3) duplicate SKUs")
sys.stdout.flush(); subprocess.run([sys.executable, "-I", "scripts/dedupe.py"], check=True)
sales = json.load(open("data/sales_2026_by_key.json"))
pairs = [p for p, s in json.load(open("data/dup_pairs.json")) if s >= 0.9]
med = sorted(v["bottles"] for v in sales.values())[len(sales) // 2]
both = [(a, b) for a, b in pairs if a in sales and b in sales]
split = [(a, b) for a, b in both if min(sales[a]["bottles"], sales[b]["bottles"]) < med <= sales[a]["bottles"] + sales[b]["bottles"]]
print(f"   sim>=0.9 pairs {len(pairs)} | both sold in 2026 {len(both)} | a half below median ({med}) but merged above: {len(split)}")
