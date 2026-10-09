"""Score LLM rerank output (hpc/out_*.jsonl) against the same candidates the review queue shows.
Policy tested: overlap gate first (margin >= 0.3 auto-accepts); for lines it sends to review,
auto-accept the LLM's pick only if its token-probability confidence >= t."""
import json, sys
path = sys.argv[1] if len(sys.argv) > 1 else "hpc/out_qwen7b.jsonl"
q = {x["id"]: x for x in json.load(open("docs/queue.json"))["items"]}
llm = {r["id"]: r for r in map(json.loads, open(path))}
n = len(q)
ok = lambda x, r: 1 <= r["choice"] <= len(x["cands"]) and x["cands"][r["choice"] - 1]["ok"]

print(f"{path}: {len(llm)} answers")
print(f"  top-1 over all lines: overlap pick {sum(x['pick_ok'] for x in q.values())/n:.1%}   LLM {sum(ok(q[i], r) for i, r in llm.items())/n:.1%}")
print(f"  LLM answered 'none' on {sum(r['choice'] == 0 for r in llm.values())} lines (every line has a true match among candidates for "
      f"{sum(any(c['ok'] for c in x['cands']) for x in q.values())}/{n})")
nones = sorted(r["conf"] for r in llm.values() if r["choice"] == 0)
wrong_none = sum(any(c["ok"] for c in q[r["id"]]["cands"]) for r in llm.values() if r["choice"] == 0)
print(f"  'none' answers: {wrong_none}/{len(nones)} had the true SKU among the candidates; "
      f"probability >= 0.9 on {sum(c >= 0.9 for c in nones)}/{len(nones)}, median {nones[len(nones)//2]:.3f}")
rev = [i for i, x in q.items() if not x["auto"]]
print(f"\n  on the {len(rev)} review lines: overlap pick right {sum(q[i]['pick_ok'] for i in rev)}   LLM right {sum(ok(q[i], llm[i]) for i in rev)}")
auto = [i for i, x in q.items() if x["auto"]]
base_ok = sum(q[i]["pick_ok"] for i in auto)
print("\n  overlap gate + LLM on review lines (auto-accept LLM pick if conf >= t)")
for t in (0.9, 0.95, 0.99, 0.999):
    extra = [i for i in rev if llm[i]["conf"] >= t and llm[i]["choice"] >= 1]
    good = base_ok + sum(ok(q[i], llm[i]) for i in extra)
    tot = len(auto) + len(extra)
    print(f"    t={t:<6} auto-accept {tot/n:6.1%}   precision {good/tot:6.1%}   still to review {n-tot}   (LLM added {len(extra)}, {sum(ok(q[i], llm[i]) for i in extra)} right)")

# Forced choice: drop the "0 = none" option and renormalize over candidates (re-scored from saved logprobs, no new run).
# "None of these" stays a human decision in the review queue.
def forced(r):
    p = {int(k): v for k, v in r["probs"].items() if int(k) >= 1}
    z = sum(p.values())
    if not z: return {**r, "choice": -1, "conf": 0.0}
    c = max(p, key=p.get)
    return {**r, "choice": c, "conf": p[c] / z}
llm = {i: forced(r) for i, r in llm.items()}
print(f"\nFORCED CHOICE  top-1 over all lines: LLM {sum(ok(q[i], r) for i, r in llm.items())/n:.1%}   "
      f"on review lines: LLM right {sum(ok(q[i], llm[i]) for i in rev)} vs overlap {sum(q[i]['pick_ok'] for i in rev)}")
for t in (0.9, 0.99, 0.999):
    extra = [i for i in rev if llm[i]["conf"] >= t and llm[i]["choice"] >= 1]
    good = base_ok + sum(ok(q[i], llm[i]) for i in extra)
    tot = len(auto) + len(extra)
    print(f"    t={t:<6} auto-accept {tot/n:6.1%}   precision {good/tot:6.1%}   still to review {n-tot}   (LLM added {len(extra)}, {sum(ok(q[i], llm[i]) for i in extra)} right)")
