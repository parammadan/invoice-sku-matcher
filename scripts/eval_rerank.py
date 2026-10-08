"""BM25(size-filtered) top-5 -> LLM rerank -> confidence gate. Usage: eval_rerank.py [--dry]"""
import json, sys
sys.path.insert(0, "scripts")
from matcher import BM25Matcher
import rerank

dry = "--dry" in sys.argv
gold = [json.loads(l) for l in open("data/gold.jsonl")]
m = BM25Matcher("data/catalog.csv")
name_of = {r["key"]: r["name"] for r in m.rows}

def correct(key, target):  # exact SKU, or a text-identical duplicate SKU of the same size
    return key == target or (key and name_of[key] == name_of[target] and key.split("|", 1)[1] == target.split("|", 1)[1])

results = []
for g in gold:
    cands = m.match(g["query"], g["pack"], g["volume_ml"], k=5)
    out = rerank.call(g["query"], cands, dry=dry)
    pick = cands[out["choice"] - 1][0] if 0 < out["choice"] <= len(cands) else None
    results.append({**g, "pick": pick, "conf": out["confidence"], "reason": out["reason"],
                    "ok": correct(pick, g["target"]), "bm25_ok": correct(cands[0][0], g["target"])})

if dry:
    print(f"dry run OK: {len(results)} prompts built. Example:\n\n{rerank.prompt(gold[1]['query'], m.match(gold[1]['query'], gold[1]['pack'], gold[1]['volume_ml']))}")
    sys.exit()

n = len(results)
print(f"BM25+size equiv@1 {sum(r['bm25_ok'] for r in results)/n:.1%}   ->   +LLM rerank equiv@1 {sum(r['ok'] for r in results)/n:.1%}")
print(f"LLM said 'no match' on {sum(r['pick'] is None for r in results)} queries (all have a true match)\n")
print("confidence gate: auto-accept if conf >= t, else human review")
for t in (0.5, 0.7, 0.8, 0.9, 0.95):
    acc = [r for r in results if r["pick"] and r["conf"] >= t]
    p = sum(r["ok"] for r in acc) / len(acc) if acc else 0
    print(f"  t={t:<4}  auto-accept {len(acc)/n:6.1%}   precision {p:6.1%}   to review {n-len(acc)}")
json.dump(results, open("data/rerank_results.json", "w"), indent=1)
