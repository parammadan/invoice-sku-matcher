"""Build the catalog + real-drift gold set from Iowa product extracts.

Catalog: one row per product key (item_no, pack, volume_ml), named by its most recent description.
Gold: every *other* description seen for that key in 2024-26 becomes a query whose answer is the key.
"""
import csv, json, collections

YEARS = (2024, 2025, 2026)
names = collections.defaultdict(list)   # key -> [(year, desc)]
meta = {}
for y in YEARS:
    for r in csv.DictReader(open(f"data/products_{y}.csv")):
        key = f'{r["item_no"]}|{r["pack"]}|{r["volume_ml"]}'
        names[key].append((y, r["desc"]))
        meta[key] = r  # latest year wins

catalog = []
with open("data/catalog.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["key", "name", "pack", "volume_ml", "vendor", "category"])
    for key, seen in names.items():
        canon = seen[-1][1]
        r = meta[key]
        w.writerow([key, canon, r["pack"], r["volume_ml"], r["vendor"], r["category"]])
        catalog.append((key, canon))

n = 0
with open("data/gold.jsonl", "w") as f:
    for key, seen in names.items():
        canon = seen[-1][1]
        for q in sorted({d for _, d in seen} - {canon}):
            item, pack, vol = key.split("|")
            f.write(json.dumps({"query": q, "pack": pack, "volume_ml": vol, "target": key}) + "\n")
            n += 1

# Same name + same size under different item numbers = truly ambiguous in the catalog
dup = collections.Counter((name, k.split("|", 1)[1]) for k, name in catalog)
print(f"catalog {len(catalog)} keys | gold {n} queries | {sum(1 for c in dup.values() if c > 1)} name+size collisions")
