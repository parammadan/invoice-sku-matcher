"""Which catalog keys are the same sellable product?

1. "USE CODE n" in a name = retired SKU redirecting to item n (same size) -> follow the redirect.
2. Identical name + identical size under different item numbers -> text can't separate them.
"""
import re

def build(rows):
    name_of = {r["key"]: r["name"] for r in rows}
    keys = set(name_of)
    def canon(k):
        seen = set()
        while k not in seen:
            seen.add(k)
            m = re.search(r"USE CODE (\d+)", name_of[k])
            nxt = f"{m.group(1)}|{k.split('|', 1)[1]}" if m else None
            if nxt in keys: k = nxt
            else: break
        return k
    def same(a, b):
        if a is None: return False
        ca, cb = canon(a), canon(b)
        return ca == cb or (name_of[ca] == name_of[cb] and ca.split("|", 1)[1] == cb.split("|", 1)[1])
    return same, canon
