"""Free reranker: IDF-weighted symmetric token overlap.

sim(q, c) = sum idf(shared tokens) / sum idf(all tokens in q or c)
Extra words on EITHER side cost you (RASPBERRY, 4YR), weighted by rarity, so frequent catalog noise
(SOOH, PET, DISCO) is cheap without hand-written rules. Confidence = top-1 sim minus the best sim of a
*different* product: candidates whose cleaned names are identical (duplicate SKUs) collapse to one,
or a product would be "unsure" between itself and its own twin.
"""
from matcher import tokens, index_name

def rerank(m, query, cands):
    q = set(tokens(query))
    idf = lambda t: m.idf.get(t, max(m.idf.values()))  # unseen token = rarest
    scored, seen = [], set()
    for key, name, bm in cands:
        clean = index_name(name)
        if clean in seen: continue
        seen.add(clean)
        c = set(tokens(clean))
        union = sum(idf(t) for t in q | c)
        scored.append((sum(idf(t) for t in q & c) / union if union else 0.0, key, name))
    scored.sort(reverse=True)
    margin = scored[0][0] - (scored[1][0] if len(scored) > 1 else 0.0)
    return scored, margin
