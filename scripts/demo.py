"""Terminal demo for the walkthrough video: three real invoice lines, before and after the fix.

    python3 -I scripts/demo.py --before   # keyword search: no size filter, redirect notes left in names
    python3 -I scripts/demo.py --after    # size/pack hard filter + redirect notes stripped, then headline results
    python3 -I scripts/demo.py --checks   # saved pgvector + 7B LLM results
"""
import json, re, sys, time
sys.path.insert(0, "scripts")
import matcher
from matcher import BM25Matcher
from equiv import build

G, R, D, B, C, X = "\033[32m", "\033[31m", "\033[2m", "\033[1m", "\033[36m", "\033[0m"
if "--checks" in sys.argv:
    G, R, D, B, C, X = "\033[32m", "\033[31m", "\033[2m", "\033[1m", "\033[36m", "\033[0m"
    pg = open("results/pgvector.txt").read().splitlines()
    print(f"{C}Postgres 16 + pgvector  (results/pgvector.txt){X}")
    for l in pg[1:]:
        name, rest = l.split(":", 1)
        top1 = re.search(r"top-1 (\S+)", rest).group(1)
        short = re.search(r"short of 5 results (\d+)", rest)
        ms = re.search(r"([\d.]+) ms/query", rest).group(1)
        name = name.split("(")[0].replace("iterative_scan=", "iter ").strip()
        warn = f"{R}{short.group(1)} queries short of 5{X}" if short and short.group(1) != "0" else f"{D}none short{X}"
        print(f"  {name:28} top-1 {top1:>6}  {ms:>4} ms  {warn}")
        time.sleep(0.6)
    llm = open("results/llm_qwen7b.txt").read()
    print(f"\n{C}Qwen2.5-7B on one A100  (results/llm_qwen7b.txt){X}")
    print(f"  allowed 'none':  {R}{re.search(r'none' + chr(39) + r' on (\d+)', llm).group(1)} 'none' answers{X}, "
          f"{re.search(r'(\d+/\d+) had the true SKU', llm).group(1)} wrong")
    time.sleep(0.6)
    print(f"  forced choice:   top-1 {B}{re.search(r'FORCED CHOICE  top-1 over all lines: LLM (\S+)', llm).group(1)}{X}  {D}(best picker){X}")
    time.sleep(0.6)
    for t, p in re.findall(r"t=(0\.9\S*)\s+auto-accept\s+\S+\s+precision\s+(\S+)", llm.split("FORCED")[1]):
        print(f"    gate on its confidence >= {t:<6} precision {R}{p}{X}")
        time.sleep(0.4)
    sys.exit()

LINES = ["ELIJAH  CRAIG TOASTED BARREL", "SOOH GRAN CENTENARIO REPOSADO", "FLOR DE CANA 12YR CENTENARIO"]
before = "--before" in sys.argv
if before:  # the original index: names indexed exactly as the catalog spells them
    matcher.index_name = lambda name: name
gold = {g["query"]: g for g in map(json.loads, open("data/gold.jsonl"))}
m = BM25Matcher("data/catalog.csv")
same, _ = build(m.rows)
row = {r["key"]: r for r in m.rows}
size = lambda k: f'{row[k]["pack"]} x {row[k]["volume_ml"]} ml'

for i, line in enumerate(LINES):
    g = gold[line]
    t = g["target"]
    print(f"{B}invoice{X}  {' '.join(line.split())}  {D}· {g['pack']} x {g['volume_ml']} ml{X}")
    time.sleep(0.6)
    top = (m.match(line, k=1) if before and i < 2 else m.match(line, g["pack"], g["volume_ml"], k=1))[0][0]
    ok = same(top, t)
    name = row[top]["name"].replace(" USE CODE", "  USE CODE")[:46]
    why = "" if ok else ("  wrong size" if row[top]["name"] == row[t]["name"] else "  wrong product")
    print(f"  {G + '✓' if ok else R + '✗'}{X} {name}  {D}· {size(top)}{X}{R}{why}{X}\n")
    time.sleep(0.9)

if not before:
    rep = open("results/report.txt").read().splitlines()
    print(f"{C}all 386 real lines  (results/report.txt){X}")
    for l in rep:
        if l.strip().startswith(("BM25, text only", "BM25 + size filter", "embeddings + size filter")):
            label, t1, t5, ws = re.match(r"\s*(.+?)\s+top-1\s+(\S+)\s+in top-5\s+(\S+)\s+wrong-size top-1\s+(\S+)", l).groups()
            print(f"  {label:24} top-1 {B}{t1:>6}{X}  top-5 {t5:>6}  wrong size {ws}")
            time.sleep(0.5)
