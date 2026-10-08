"""Flag catalog SKUs that are probably the same sellable product.

Within each (pack, volume) group, score every pair by IDF-weighted symmetric token overlap.
Redirect notes ("USE CODE n", "USE n") are stripped BEFORE scoring, so the catalog's own
redirects act as labeled true duplicates we can measure recall against.
"""
import csv, re, math, json, sys, collections, itertools
sys.path.insert(0, "scripts")
from matcher import tokens

REDIRECT = re.compile(r"-?\s*USE\s+(?:CODE\s+)?(\d+)")
rows = list(csv.DictReader(open("data/catalog.csv")))
clean = {r["key"]: REDIRECT.sub("", r["name"]).strip() for r in rows}
toks = {k: set(tokens(v)) for k, v in clean.items()}
df = collections.Counter(t for s in toks.values() for t in s)
N = len(rows)
idf = {t: math.log(1 + (N - c + 0.5) / (c + 0.5)) for t, c in df.items()}
w = lambda s: sum(idf[t] for t in s)

# labeled positives: X says "USE CODE n" and item n exists at the same size
by_item_size = {(r["key"].split("|")[0], r["key"].split("|", 1)[1]): r["key"] for r in rows}
labeled = set()
for r in rows:
    m = REDIRECT.search(r["name"])
    tgt = m and by_item_size.get((m.group(1), r["key"].split("|", 1)[1]))
    if tgt and tgt != r["key"]: labeled.add(frozenset((r["key"], tgt)))

groups = collections.defaultdict(list)
for r in rows: groups[(r["pack"], r["volume_ml"])].append(r["key"])
pairs = {}
for keys in groups.values():
    for a, b in itertools.combinations(keys, 2):
        A, B = toks[a], toks[b]
        if A & B:
            s = w(A & B) / w(A | B)
            if s >= 0.5: pairs[frozenset((a, b))] = s

print(f"{len(labeled)} labeled redirect pairs (true duplicates)")
for t in (0.6, 0.7, 0.8, 0.9, 1.0):
    flagged = {p for p, s in pairs.items() if s >= t - 1e-9}
    rec = len(flagged & labeled) / len(labeled)
    print(f"  sim>={t}: flagged {len(flagged):5d} pairs   recall on redirects {rec:6.1%}")

json.dump(sorted(([*p], s) for p, s in pairs.items()), open("data/dup_pairs.json", "w"))
