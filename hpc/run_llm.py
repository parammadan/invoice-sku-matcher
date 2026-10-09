"""vLLM batch rerank. One output token, temperature 0; confidence = softmax over candidate-digit logprobs.
Usage: python run_llm.py MODEL prompts.jsonl out.jsonl
Loads the model once, runs a 5-prompt smoke batch, and exits non-zero if that output is malformed,
before spending GPU time on the full batch."""
import json, math, sys
from vllm import LLM, SamplingParams

model, src, dst = sys.argv[1:4]
rows = [json.loads(l) for l in open(src)]
llm = LLM(model=model, dtype="auto", max_model_len=2048, gpu_memory_utilization=0.85)
sp = SamplingParams(temperature=0, max_tokens=1, logprobs=20)

def parse(r, o):
    lp = o.outputs[0].logprobs[0]
    digits = {}
    for tok_id, info in lp.items():
        t = (info.decoded_token or "").strip()
        if t.isascii() and t.isdigit() and int(t) <= r["n"]:  # skip look-alikes such as subscript digits
            digits[int(t)] = max(digits.get(int(t), -1e9), info.logprob)
    z = sum(math.exp(v) for v in digits.values()) or 1
    probs = {k: math.exp(v) / z for k, v in digits.items()}
    choice = max(probs, key=probs.get) if probs else -1
    return {"id": r["id"], "text": o.outputs[0].text, "choice": choice, "conf": probs.get(choice, 0.0), "probs": probs}

smoke = [parse(r, o) for r, o in zip(rows[:5], llm.chat([r["messages"] for r in rows[:5]], sp))]
print("SMOKE", smoke, flush=True)
if not all(x["choice"] >= 0 for x in smoke):
    sys.exit("SMOKE FAILED: no digit answer")
res = [parse(r, o) for r, o in zip(rows, llm.chat([r["messages"] for r in rows], sp))]
with open(dst, "w") as f:
    for x in res: f.write(json.dumps(x) + "\n")
print("RESULT", len(res), "rows written", dst)
