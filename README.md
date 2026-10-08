# Invoice line → SKU matching for liquor retail

A small, free, fully reproducible study of the matching step behind invoice receiving:
given a distributor line like `SOOH FLOR DE CANA 12YR CENTENARIO`, find the store's catalog SKU,
**at the right size**, and know when to ask a human instead.

Everything runs on a laptop: Python's standard library plus a local [Ollama](https://ollama.com) embedding model. No API keys, no paid models.

## Data (all real)

[Iowa Liquor Sales](https://data.iowa.gov/catalog/dataset/1263), 2024–2026, state open data:
every spirits purchase by Iowa retailers, with item number, description, pack and bottle size.

- **Catalog:** 7,372 SKUs, keyed on item number + pack + volume and named by the latest description.
- **Gold set:** 386 real queries. When the same SKU appears under a different description in another
  year, that old description becomes a query whose answer is the SKU. Nothing is synthetic.
  The noise is real: status prefixes (`SOOH`, `CM`, `HA`), `DISCO`, `PET`, `USE CODE 42631`, renames
  (`TEMPLETON RYE 4YR` → `TEMPLETON STRAIGHT RYE`).
- **The size trap is real too:** 412 descriptions exist at more than one size. `TITOS HANDMADE VODKA`
  comes in 7 pack/volume combinations.

Scoring is **redirect-aware**: a name containing `USE CODE n` is a retired SKU pointing to item `n`,
so it counts as the same product. SKUs with identical text and size count as equivalent,
because no text matcher could separate them.

## Results

`python3 -I scripts/report.py` reproduces every number below.

### 1. Matching

| | Correct top-1 | Right answer in top 5 | Wrong-size top-1 |
|---|---|---|---|
| BM25, text only | 78.2% | 96.9% | 2.1% |
| BM25 + hard size/pack filter | 89.1% | 97.9% | **0.0%** |
| **Embeddings + size filter** (nomic-embed-text, local) | **89.9%** | **99.0%** | **0.0%** |
| Hybrid, reciprocal rank fusion + size filter | 89.6% | 97.9% | 0.0% |

- **Filtering on pack and volume before ranking is the biggest single win.** It also removes the
  most expensive error: right product, wrong size. That error silently corrupts cost per bottle and margin alerts.
- **BM25 and embeddings fail on different lines.** Embeddings catch meaning: the real rebrand
  `PLANTATION RUM` → `PLANTERAY RUM`, `DR` → `DOCTOR`, `CLASSIC` → `ORIGINAL`. BM25 catches exact
  tokens embeddings blur: `OLD FORESTER 1920`, `ZONE VODKA`. 9 lines only BM25 gets right,
  12 only embeddings get right. Naive RRF fusion doesn't beat embeddings alone at top-1.
- **Redirect notes are stripped before indexing.** `FLOR DE CANA 12YR … USE CODE 42631` lost to
  the shorter `FLOR DE CANA 25YR` because the note's three tokens triggered BM25's length penalty. Wrong age,
  far more expensive bottle. The note is metadata, not the name.

### 2. Confidence gate: when to skip the human

Candidates are re-scored by IDF-weighted symmetric token overlap. Extra words on either side
(`RASPBERRY`, `15YR`) cost a lot; frequent catalog noise (`SOOH`, `PET`) costs little, with no hand
rules. The margin between candidates 1 and 2 is the confidence.

| Gate | Auto-accepted | Precision of auto-accepts | Sent to review |
|---|---|---|---|
| none | 100% | 89.4% | 0 |
| margin ≥ 0.2 | 51.8% | 97.0% | 186 |
| **margin ≥ 0.3** | **41.5%** | **99.4%** | 226 |
| BM25 and embeddings agree | 94.3% | 92.6% | 22 |

**41.5% of lines need no human, and those are right 99.4% of the time.** The rest go to a review
queue with candidates already ranked: embeddings put the right answer in the top 5 for 99.0% of lines.

Agreement between BM25 and embeddings sounds like a good confidence signal but isn't: when both
are wrong, they're usually wrong *together*, picking the same text-twin of a duplicate SKU (section 3).

### 3. Duplicate SKUs

Most residual "errors" turned out to be **one bottle listed under two SKUs**, not matcher mistakes:
`99 GRAPES` vs `SOOH 99 GRAPES`, `NELSON BROS.` vs `NELSON BROS`.

`scripts/dedupe.py` scores every same-size SKU pair. The catalog's own `USE CODE` redirects are
labeled true duplicates, and the redirect text is **stripped before scoring**, so recall is measured
honestly:

| Similarity ≥ | Pairs flagged | Recall on known redirects |
|---|---|---|
| 1.0 | 294 | 60.0% |
| 0.9 | 303 | 60.0% |
| 0.8 | 483 | 68.9% |
| 0.7 | 939 | 77.8% |

Precision has no labels. From eyeballing about 10 pairs per band: 1.0 is the same product, 0.8–1.0
is mostly the same with traps (`ANEJO` vs `EXTRA ANEJO`), and 0.7–0.8 is mostly **different**
(gin vs vodka, 12YR vs 15YR). Suggested policy: auto-flag at 1.0, human review for 0.8–1.0.

**Why it matters for slow-mover detection:** in 17 of the 55 high-confidence pairs where both SKUs
sold in 2026, one half is below the median SKU (492 bottles) while the merged product is above it.
For example, `PRIVATE FIRST CLASS` sold 96 + 1,883 = 1,979 bottles. Most of these look like code
transitions (the old SKU tails off as the new one ramps). A per-SKU slow-mover report would still flag
a product selling about 2,000 bottles. **Slow movers have to be computed per product, so duplicates
must be resolved first.**

### 4. In Postgres + pgvector

`scripts/pg_eval.py` loads the catalog and vectors into Postgres 16 + pgvector and runs the
size-filtered search in one SQL query (`CROSS JOIN LATERAL … WHERE pack = … AND volume_ml = …
ORDER BY embedding <=> query LIMIT 5`). It reproduces the Python numbers exactly.

| | Top-1 | Top-5 | Queries with < 5 results | ms/query |
|---|---|---|---|---|
| **Exact: btree size filter, then sort by distance** | **89.9%** | **99.0%** | 0 | 4.7 |
| HNSW, `iterative_scan = off` | 88.9% | 97.9% | **38 (10%)** | 0.3 |
| HNSW, `iterative_scan = relaxed_order` + exact re-sort | 89.4% | 98.7% | 0 | 0.4 |

**The trap:** HNSW collects `ef_search` nearest neighbours first and applies the `WHERE` after, so
with a selective filter (one pack/volume) 10% of queries **silently return fewer than 5
candidates**. That's a review queue missing its right answer, with no error raised.
pgvector 0.8's iterative scan fixes it, and `relaxed_order` needs an exact re-sort on top.
**At store-catalog size (thousands of SKUs per size bucket), exact filtered search at ~5 ms is the
right call.** HNSW only earns its place across a multi-store or industry-wide catalog.

## What didn't work

**A local 3B LLM re-ranker (Qwen2.5-3B via Ollama) made things worse.** On a 40-query sample it got
13/40 vs BM25's 31/40, answered "no match" 24 times when a match existed, and reported ≥ 0.9
confidence on all 40, so its confidence can't drive a gate. The code (`scripts/rerank.py`) supports
Ollama or the Anthropic API. A stronger model is the natural next experiment, judged against the
same gold set.

## Limitations

- **Wholesale, not retail:** Iowa data is stores buying from the state, not shoppers buying from stores.
- **Catalog descriptions, not distributor invoices:** real invoices can be messier (`TITOS HMD VDK`).
  Synthetic abbreviation noise would be a separate, clearly labeled test set.
- **The LLM prompt names noise tokens** (`SOOH`, `DISCO`) that I saw in this data, so any LLM score
  here would be optimistic without a held-out split.
- **Duplicate-pair precision is eyeballed,** not measured.
- **2026 is year-to-date** (through early October).

## Reproduce

```bash
# ~400MB per year; Iowa Data Hub dataset ids 1261/1262/1263 = 2024/2025/2026
mkdir -p data
for y in 2024:1261 2025:1262 2026:1263; do
  curl -s -o data/sales_${y%%:*}.zip "https://idh-be.iowa.gov/api/v1/datasets/${y##*:}/rows.csv"
  python3 -I scripts/extract_products.py data/sales_${y%%:*}.zip data/products_${y%%:*}.csv
done
python3 -I scripts/build_eval.py
ollama pull nomic-embed-text && python3 -I scripts/embed.py   # ~7 min on an M1, needs `ollama serve`
python3 -I scripts/sales_by_key.py data/sales_2026.zip data/sales_2026_by_key.json
python3 -I scripts/report.py

# optional: Postgres 16+ with pgvector (project-local instance on port 5433)
initdb -D data/pg -U matcher --auth=trust && pg_ctl -D data/pg -o "-p 5433 -k /tmp" -l data/pg.log start
python3 -I scripts/pg_eval.py
```

| File | What it does |
|---|---|
| `scripts/matcher.py` | BM25 + hard size/pack filter |
| `scripts/embed.py`, `scripts/hybrid.py` | local embeddings, dense search, RRF hybrid |
| `scripts/overlap_rerank.py` | IDF-weighted overlap re-score + margin confidence |
| `scripts/equiv.py` | `USE CODE` redirect resolution + equivalence for scoring |
| `scripts/dedupe.py` | duplicate-SKU detection with recall on redirects |
| `scripts/rerank.py` | optional LLM re-ranker (Ollama or Anthropic) |
| `scripts/pg_eval.py` | the same search in Postgres + pgvector: exact vs HNSW |
| `scripts/report.py` | every number in sections 1–3 |
