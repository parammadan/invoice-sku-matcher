"""Free reranker: IDF-weighted symmetric token overlap.

sim(q, c) = sum idf(shared tokens) / sum idf(all tokens in q or c)
Extra words on EITHER side cost you (RASPBERRY, 4YR), weighted by rarity, so frequent catalog noise
(SOOH, PET, DISCO) is cheap without hand-written rules. Confidence = top-1 sim minus top-2 sim.
"""
from matcher import tokens

def rerank(m, query, cands):
    q = set(tokens(query))
    idf = lambda t: m.idf.get(t, max(m.idf.values()))  # unseen token = rarest
    scored = []
    for key, name, bm in cands:
        c = set(tokens(name))
        union = sum(idf(t) for t in q | c)
        scored.append((sum(idf(t) for t in q & c) / union if union else 0.0, key, name))
    scored.sort(reverse=True)
    margin = scored[0][0] - (scored[1][0] if len(scored) > 1 else 0.0)
    return scored, margin
