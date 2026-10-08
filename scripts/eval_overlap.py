"""BM25(size-filtered) top-k -> overlap rerank -> margin gate."""
import json, sys
sys.path.insert(0, "scripts")
from matcher import BM25Matcher
from overlap_rerank import rerank

gold = [json.loads(l) for l in open("data/gold.jsonl")]
m = BM25Matcher("data/catalog.csv")
name_of = {r["key"]: r["name"] for r in m.rows}
size = lambda k: k.split("|", 1)[1]
ok = lambda k, t: k == t or (name_of[k] == name_of[t] and size(k) == size(t))

for k in (5, 10, 20):
    res = []
    for g in gold:
        cands = m.match(g["query"], g["pack"], g["volume_ml"], k=k)
        scored, margin = rerank(m, g["query"], cands)
        res.append((ok(scored[0][1], g["target"]), margin, ok(cands[0][0], g["target"]), g, scored[0][2]))
    n = len(res)
    print(f"top-{k:<2} bm25 {sum(r[2] for r in res)/n:.1%}  ->  overlap rerank {sum(r[0] for r in res)/n:.1%}")
    if k == 10: best = res

print("\nmargin gate (top-10): auto-accept if margin >= t")
for t in (0.0, 0.05, 0.1, 0.15, 0.2, 0.3):
    a = [r for r in best if r[1] >= t]
    print(f"  t={t:<4} auto-accept {len(a)/len(best):6.1%}  precision {sum(r[0] for r in a)/len(a):6.1%}  to review {len(best)-len(a)}")
print("\nconfident mistakes (margin >= 0.1):")
for r in sorted((r for r in best if not r[0] and r[1] >= 0.1), key=lambda r: -r[1])[:8]:
    print(f"  {r[1]:.2f}  Q {r[3]['query']!r}  got {r[4]!r}  want {name_of[r[3]['target']]!r}")
