"""Embedding search and BM25+embedding hybrid (reciprocal rank fusion), both size-filtered."""
import json

class Hybrid:
    def __init__(self, bm25, cat_path="data/emb_catalog.json", q_path="data/emb_queries.json"):
        self.m = bm25
        self.cat = json.load(open(cat_path))
        self.qv = json.load(open(q_path))

    def _pool(self, pack, volume_ml):
        return [self.m.rows[i]["key"] for i in self.m.by_size.get((pack, volume_ml), [])]

    def dense(self, query, pack, volume_ml, k=10, qvec=None):
        q = qvec or self.qv[query]
        scored = sorted(((sum(a * b for a, b in zip(q, self.cat[key])), key) for key in self._pool(pack, volume_ml)), reverse=True)
        return [key for _, key in scored[:k]]

    def fused(self, query, pack, volume_ml, k=10, depth=50, c=60, qvec=None):
        bm = [key for key, _, _ in self.m.match(query, pack, volume_ml, k=depth)]
        de = self.dense(query, pack, volume_ml, k=depth, qvec=qvec)
        score = {}
        for ranks in (bm, de):
            for r, key in enumerate(ranks):
                score[key] = score.get(key, 0) + 1 / (c + r + 1)
        return [key for key, _ in sorted(score.items(), key=lambda x: -x[1])[:k]]
