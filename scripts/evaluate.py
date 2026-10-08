"""Score a matcher on the real-drift gold set.

strict@1   top-1 key == target
equiv@1    top-1 has the target's exact name+size (duplicate SKU, indistinguishable by text)
wrong_size top-1 is the right product name but a different pack/volume - the costly error
"""
import json, sys, csv
sys.path.insert(0, "scripts")
from matcher import BM25Matcher

gold = [json.loads(l) for l in open("data/gold.jsonl")]
m = BM25Matcher("data/catalog.csv")
name_of = {r["key"]: r["name"] for r in m.rows}

def run(use_size):
    c = {"strict@1": 0, "equiv@1": 0, "hit@5": 0, "wrong_size": 0}
    misses = []
    for g in gold:
        res = m.match(g["query"], g["pack"], g["volume_ml"]) if use_size else m.match(g["query"])
        keys = [r[0] for r in res]
        top = keys[0] if keys else None
        tsize, tname = g["target"].split("|", 1)[1], name_of[g["target"]]
        if top == g["target"]: c["strict@1"] += 1
        if top and (top == g["target"] or (name_of[top] == tname and top.split("|", 1)[1] == tsize)): c["equiv@1"] += 1
        else: misses.append((g["query"], tname, res[0][1] if res else None))
        if g["target"] in keys: c["hit@5"] += 1
        if top and name_of[top] == tname and top.split("|", 1)[1] != tsize: c["wrong_size"] += 1
    n = len(gold)
    print(f"{'size filter' if use_size else 'text only  '}  " + "  ".join(f"{k} {v/n:6.1%}" for k, v in c.items()))
    return misses

run(False)
misses = run(True)
print(f"\n{len(misses)} misses with size filter, first 12:")
for q, t, got in misses[:12]: print(f"  Q {q!r}\n    want {t!r}\n    got  {got!r}")
