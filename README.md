# Invoice line → SKU matching for liquor retail

A small, free, fully reproducible study of the matching step behind invoice receiving:
given a distributor line like `SOOH FLOR DE CANA 12YR CENTENARIO`, find the store's catalog SKU,
**at the right size**, and know when to ask a human instead.

**[Try the review queue →](https://parammadan.github.io/invoice-sku-matcher/)**: the gate's real decisions on all 386 lines, with the true SKU revealed after you choose.

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
rules. Confidence is the margin between the top candidate and the best **different** product.

| Gate | Auto-accepted | Precision of auto-accepts | Sent to review |
|---|---|---|---|
| none | 100% | 89.1% | 0 |
| margin ≥ 0.2 | 73.1% | 97.5% | 104 |
| **margin ≥ 0.3** | **59.6%** | **99.1%** | 156 |
| margin ≥ 0.3 **and** BM25 + embeddings agree | 59.1% | 99.6% | 158 |

**60% of lines need no human, and those are right 99.1% of the time.** The rest go to a review
queue with candidates already ranked: embeddings put the right answer in the top 5 for 99.0% of lines.

**Bug worth knowing about:** the first version of the gate auto-accepted only 41.5%. The catalog
holds exact-twin SKUs (two `JINRO GREEN GRAPE SOJU` at 20 × 375 ml), so the top two candidates
were the same product and the margin was 0: the matcher looked "unsure" between a product and its
own twin. Collapsing identical names before measuring the margin took auto-accept from 41.5% to 59.6%
at about the same precision. Duplicate SKUs don't just split sales (section 3); they also make
the matcher under-confident.

Agreement between BM25 and embeddings alone is a poor gate (94% auto-accepted at 92.6%): when both
are wrong, they're usually wrong *together*, on the same text-twin of a duplicate SKU.

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

### 5. LLM re-ranking: a better picker, a badly calibrated judge

Qwen2.5-7B-Instruct, served with vLLM on one A100 (Northeastern's research cluster, about 5 minutes
of GPU time for all 386 lines). It sees the invoice line plus the review queue's 5 distinct candidates
and answers with one number. Confidence comes from the **token probabilities** of the candidate
numbers, not from asking the model to rate itself. The prompt describes catalog noise in general
terms; it names no tokens copied from this dataset. `hpc/run_llm.py`, `scripts/eval_llm.py`.

| Top-1 correct | All 386 lines | The 156 lines the gate sends to review |
|---|---|---|
| Overlap re-ranker | 89.1% | 116 |
| 7B LLM, may answer "none of these" | 75.9% | 97 |
| **7B LLM, forced choice among candidates** | **91.2%** | **124** |

- **Allowed to say "none," it says it far too often:** 69 times, with probability ≈ 1.0 on lines like
  `TYCOGA VODKA` vs `SOOH TYCOGA VODKA`. All 23 of its breaks of correct picks were "none."
  Forced choice (scored from the same saved probabilities, no second run) fixes that and becomes
  **the most accurate method here**. It catches abbreviations like `WYCH DR` → `WYCH DOCTOR`.
- **Its confidence can't drive a gate.** Auto-accepting forced-choice picks at probability ≥ 0.9,
  0.99 or 0.999 gives about 92% precision every time, because it is almost always "sure."
- **So the roles split:** the overlap margin decides what skips a human (60% at 99.1%), and the
  LLM chooses the suggestion shown to the reviewer for everything else (124/156 right vs 116).
  "None of these" stays a human call.

An earlier attempt with Qwen2.5-**3B** on a laptop (Ollama) got 13/40 vs BM25's 31/40 on a sample
and reported ≥ 0.9 self-rated confidence on every answer. Self-reported confidence was the first
thing to go.

## Limitations

- **Wholesale, not retail:** Iowa data is stores buying from the state, not shoppers buying from stores.
- **Catalog descriptions, not distributor invoices:** real invoices can be messier (`TITOS HMD VDK`).
  Synthetic abbreviation noise would be a separate, clearly labeled test set.
- **The local Ollama prompt (`scripts/rerank.py`) names noise tokens** seen in this data; the reported 7B
  results use the generic prompt in `scripts/export_llm_prompts.py` instead.
- **LLM candidates come from the matcher:** the true SKU is among the 5 for 381 of 386 lines, which caps LLM top-1.
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
| `scripts/rerank.py` | optional local LLM re-ranker (Ollama or Anthropic API) |
| `scripts/export_llm_prompts.py`, `hpc/run_llm.py`, `scripts/eval_llm.py` | 7B vLLM batch re-rank + scoring |
| `scripts/pg_eval.py` | the same search in Postgres + pgvector: exact vs HNSW |
| `scripts/build_queue.py`, `docs/` | the review-queue page (static, GitHub Pages) |
| `scripts/report.py` | every number in sections 1–3 |
