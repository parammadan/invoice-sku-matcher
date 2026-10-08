"""Bottles sold + number of stores per product key, from one year's Iowa sales zip."""
import csv, sys, zipfile, io, collections, json
zp, out = sys.argv[1], sys.argv[2]
bottles, stores = collections.Counter(), collections.defaultdict(set)
with zipfile.ZipFile(zp) as z:
    for name in sorted(z.namelist()):
        with z.open(name) as f:
            for r in csv.DictReader(io.TextIOWrapper(f, encoding="utf-8", errors="replace")):
                k = f'{r["item_no"]}|{r["pack"]}|{r["bottle_volume_ml"]}'
                try: bottles[k] += int(float(r["sales_bottles"] or 0))
                except ValueError: pass
                stores[k].add(r["store_no"])
json.dump({k: {"bottles": bottles[k], "stores": len(stores[k])} for k in stores}, open(out, "w"))
print(len(stores), "keys with 2026 sales")
