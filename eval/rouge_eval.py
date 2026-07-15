"""Clean ROUGE eval: base unsloth/Llama-3.2-3B vs fine-tuned seasalt29/model3.

Eval data: Amod/mental_health_counseling_conversations (CounselChat-derived,
real counselor answers) — NOT part of MentalChat16K training data.
Both models get the exact alpaca prompt template used during fine-tuning
(from Model3.ipynb) with the same counseling instruction, greedy decoding.
Models are run one at a time (16 GB RAM), sequentially.
"""
import gc
import json
import random
import sys
import time

import torch
from datasets import load_dataset
from rouge_score import rouge_scorer
from transformers import AutoModelForCausalLM, AutoTokenizer

N_SAMPLES = 100
SEED = 3407  # same seed the training notebook used
MAX_NEW_TOKENS = 256
INSTRUCTION = (
    "You are a helpful mental health counselling assistant, please answer the "
    "mental health questions based on the patient's description. The assistant "
    "gives helpful, comprehensive, and appropriate answers to the user's questions."
)
ALPACA_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

OUT_PATH = sys.argv[1] if len(sys.argv) > 1 else "elyssa_eval_results.json"

print("Loading eval dataset...", flush=True)
ds = load_dataset("Amod/mental_health_counseling_conversations", split="train")
random.seed(SEED)
idx = random.sample(range(len(ds)), N_SAMPLES)
samples = [(ds[i]["Context"], ds[i]["Response"]) for i in idx]
print(f"dataset size {len(ds)}, sampled {len(samples)} with seed {SEED}", flush=True)


def generate_all(model_id):
    print(f"Loading {model_id} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16)
    model = model.to("mps").eval()
    outs = []
    t0 = time.time()
    for n, (ctx, _ref) in enumerate(samples):
        prompt = ALPACA_PROMPT.format(INSTRUCTION, ctx, "")
        inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=1024).to("mps")
        with torch.no_grad():
            gen = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
            )
        text = tok.decode(gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        outs.append(text.strip())
        if (n + 1) % 10 == 0:
            print(f"  {model_id}: {n+1}/{N_SAMPLES} ({time.time()-t0:.0f}s)", flush=True)
    del model
    gc.collect()
    torch.mps.empty_cache()
    return outs


def rouge_f1(preds):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    sums = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    for pred, (_ctx, ref) in zip(preds, samples):
        s = scorer.score(ref, pred)
        for k in sums:
            sums[k] += s[k].fmeasure
    return {k: v / len(preds) for k, v in sums.items()}


results = {"n_samples": N_SAMPLES, "seed": SEED, "max_new_tokens": MAX_NEW_TOKENS,
           "eval_dataset": "Amod/mental_health_counseling_conversations",
           "decoding": "greedy", "models": {}}

for model_id in ["unsloth/Llama-3.2-3B", "seasalt29/model3"]:
    preds = generate_all(model_id)
    scores = rouge_f1(preds)
    results["models"][model_id] = {"rouge_f1": scores,
                                   "sample_outputs": preds[:3]}
    print(model_id, "->", scores, flush=True)

base = results["models"]["unsloth/Llama-3.2-3B"]["rouge_f1"]
ft = results["models"]["seasalt29/model3"]["rouge_f1"]
results["relative_improvement_pct"] = {
    k: (ft[k] - base[k]) / base[k] * 100 if base[k] else None for k in base
}
print("relative improvement %:", results["relative_improvement_pct"], flush=True)

with open(OUT_PATH, "w") as f:
    json.dump(results, f, indent=2)
print("wrote", OUT_PATH, flush=True)
