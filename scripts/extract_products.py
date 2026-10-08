import csv, sys, zipfile, io, collections
zp, out = sys.argv[1], sys.argv[2]
seen = {}
with zipfile.ZipFile(zp) as z:
    for name in sorted(z.namelist()):
        with z.open(name) as f:
            for r in csv.DictReader(io.TextIOWrapper(f, encoding="utf-8", errors="replace")):
                key = (r["item_no"], r["im_desc"].strip(), r["pack"], r["bottle_volume_ml"])
                if key not in seen:
                    seen[key] = (r["vendor_name"], r["category_name"], r["state_bottle_cost"])
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["item_no","desc","pack","volume_ml","vendor","category","bottle_cost"])
    for k, v in seen.items(): w.writerow([*k, *v])
print(len(seen), "unique rows;", len({k[0] for k in seen}), "item_nos")
