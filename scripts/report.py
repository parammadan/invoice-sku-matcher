"""Reproduce every number in the README. Run from repo root after the data steps."""
import json, sys, subprocess
sys.path.insert(0, "scripts")
from matcher import BM25Matcher
from overlap_rerank import rerank
from equiv import build

gold = [json.loads(l) for l in open("data/gold.jsonl")]
m = BM25Matcher("data/catalog.csv")
same, _ = build(m.rows)
name_of = {r["key"]: r["name"] for r in m.rows}
n = len(gold)
print(f"catalog {len(m.rows)} SKUs | gold {n} real queries\n")

print("1) matching (redirect-aware scoring)")
for label, sized in (("text only", False), ("+ size/pack filter", True)):
    top1 = hit5 = wrong = 0
    for g in gold:
        c = m.match(g["query"], g["pack"], g["volume_ml"]) if sized else m.match(g["query"])
        top1 += same(c[0][0], g["target"])
        hit5 += any(same(k, g["target"]) for k, _, _ in c)
        wrong += (not same(c[0][0], g["target"])) and name_of[c[0][0]] == name_of[g["target"]] and c[0][0].split("|", 1)[1] != g["target"].split("|", 1)[1]
    print(f"   {label:20} top-1 {top1/n:6.1%}   in top-5 {hit5/n:6.1%}   wrong-size top-1 {wrong/n:5.1%}")

print("\n2) confidence gate (overlap margin on size-filtered top-10)")
res = []
for g in gold:
    s, margin = rerank(m, g["query"], m.match(g["query"], g["pack"], g["volume_ml"], k=10))
    res.append((same(s[0][1], g["target"]), margin))
for t in (0.0, 0.1, 0.2, 0.3):
    a = [ok for ok, mg in res if mg >= t]
    print(f"   margin >= {t}: auto-accept {len(a)/n:6.1%}   precision {sum(a)/len(a):6.1%}   to review {n-len(a)}")

print("\n3) duplicate SKUs")
sys.stdout.flush(); subprocess.run([sys.executable, "-I", "scripts/dedupe.py"], check=True)
sales = json.load(open("data/sales_2026_by_key.json"))
pairs = [p for p, s in json.load(open("data/dup_pairs.json")) if s >= 0.9]
med = sorted(v["bottles"] for v in sales.values())[len(sales) // 2]
both = [(a, b) for a, b in pairs if a in sales and b in sales]
split = [(a, b) for a, b in both if min(sales[a]["bottles"], sales[b]["bottles"]) < med <= sales[a]["bottles"] + sales[b]["bottles"]]
print(f"   sim>=0.9 pairs {len(pairs)} | both sold in 2026 {len(both)} | a half below median ({med}) but merged above: {len(split)}")
