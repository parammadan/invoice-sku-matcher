"""Baseline matcher: BM25 over catalog names, optionally hard-filtered on pack + volume."""
import csv, math, re, collections

# "USE CODE 42631" / "- USE 25776" are catalog redirect notes, not part of the product name
REDIRECT = re.compile(r"-?\s*USE\s+(?:CODE\s+)?\d+")

def index_name(name):
    return REDIRECT.sub("", name).strip()

def tokens(s):
    return re.findall(r"[a-z0-9]+", s.lower())

class BM25Matcher:
    def __init__(self, catalog_path, k1=1.2, b=0.75):
        self.rows = list(csv.DictReader(open(catalog_path)))
        self.docs = [tokens(index_name(r["name"])) for r in self.rows]
        self.k1, self.b = k1, b
        self.avgdl = sum(map(len, self.docs)) / len(self.docs)
        df = collections.Counter(t for d in self.docs for t in set(d))
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}
        self.tf = [collections.Counter(d) for d in self.docs]
        self.by_size = collections.defaultdict(list)
        for i, r in enumerate(self.rows):
            self.by_size[(r["pack"], r["volume_ml"])].append(i)

    def score(self, q, i):
        tf, dl, s = self.tf[i], len(self.docs[i]), 0.0
        for t in q:
            if t in tf:
                f = tf[t]
                s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
        return s

    def match(self, query, pack=None, volume_ml=None, k=5):
        q = tokens(query)
        pool = self.by_size.get((pack, volume_ml), []) if pack else range(len(self.rows))
        scored = sorted(((self.score(q, i), i) for i in pool), reverse=True)[:k]
        return [(self.rows[i]["key"], self.rows[i]["name"], s) for s, i in scored]
