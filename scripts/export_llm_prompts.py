"""Write hpc/prompts.jsonl: every gold line with the same distinct candidates the review queue shows.
The prompt describes catalog noise generically (no tokens copied from this dataset) to avoid tuning on the test set."""
import json
SYSTEM = ("You match lines from liquor distributor invoices to a store's product catalog. Every candidate already "
          "has the same bottle size and pack. Catalog names can carry status prefixes, discontinuation or packaging "
          "notes, and abbreviations; ignore those. Different flavors, ages, proofs, expressions, editions and gift sets "
          "are DIFFERENT products. Reply with only the number of the candidate that is the same product, or 0 if none is.")
q = json.load(open("docs/queue.json"))
with open("hpc/prompts.jsonl", "w") as f:
    for x in q["items"]:
        user = f"Invoice line: {x['line']}\n\nCandidates:\n" + "\n".join(f"{i}. {c['name']}" for i, c in enumerate(x["cands"], 1))
        f.write(json.dumps({"id": x["id"], "n": len(x["cands"]), "messages": [
            {"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]}) + "\n")
print(sum(1 for _ in open("hpc/prompts.jsonl")), "prompts")
