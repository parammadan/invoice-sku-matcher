"""Export the review-queue page data: docs/queue.json (gate decisions, candidates, duplicate pairs)."""
import json, sys
sys.path.insert(0, "scripts")
from matcher import BM25Matcher, index_name
from hybrid import Hybrid
from overlap_rerank import rerank
from equiv import build

GATE = 0.3
gold = [json.loads(l) for l in open("data/gold.jsonl")]
m = BM25Matcher("data/catalog.csv"); h = Hybrid(m); same, _ = build(m.rows)
nm = {r["key"]: r["name"] for r in m.rows}
items = []
for i, g in enumerate(gold):
    s, margin = rerank(m, g["query"], m.match(g["query"], g["pack"], g["volume_ml"], k=10))
    pick = s[0][1]
    cands, names = [], set()
    for k in [pick] + h.dense(g["query"], g["pack"], g["volume_ml"], k=12):  # distinct names only
        if index_name(nm[k]) not in names and not (cands and same(k, pick)):
            cands.append(k); names.add(index_name(nm[k]))
    cands = cands[:5]
    items.append({
        "id": i, "line": g["query"], "pack": int(g["pack"]), "ml": int(g["volume_ml"]),
        "margin": round(margin, 3), "auto": margin >= GATE,
        "cands": [{"key": k, "name": index_name(nm[k]), "ok": same(k, g["target"])} for k in cands],
        "pick_ok": same(pick, g["target"]), "truth": index_name(nm[g["target"]]),
    })
sales = json.load(open("data/sales_2026_by_key.json"))
dups = []
for (a, b), sim in json.load(open("data/dup_pairs.json")):
    if sim >= 0.8:
        dups.append({"a": nm[a], "b": nm[b], "size": a.split("|", 1)[1].replace("|", " × ") + " ml", "sim": round(sim, 2),
                     "sa": sales.get(a, {}).get("bottles", 0), "sb": sales.get(b, {}).get("bottles", 0)})
dups.sort(key=lambda d: (-d["sim"], -(d["sa"] + d["sb"])))
json.dump({"gate": GATE, "items": items, "dups": dups}, open("docs/queue.json", "w"), separators=(",", ":"))
auto = [x for x in items if x["auto"]]
print(f"{len(items)} lines: {len(auto)} auto ({sum(x['pick_ok'] for x in auto)/len(auto):.1%} correct), "
      f"{len(items)-len(auto)} review | {len(dups)} dup pairs (sim>=0.8)")
