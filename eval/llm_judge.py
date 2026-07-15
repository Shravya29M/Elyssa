"""Blind pairwise LLM-judge: base Llama-3.2-3B vs fine-tuned seasalt29/model3.

Reads the full generations from elyssa_eval_full_results.json, presents each
pair to an OpenAI judge model (key from OPENAI_API_KEY env var) with randomized
A/B order to control position bias. Judge picks A, B, or TIE.
"""
import json
import os
import random
import urllib.request

RESULTS = "generations.json"
OUT = "judge_results.json"
SEED = 3407
JUDGE_MODEL = "gpt-5.1"

JUDGE_PROMPT = """You are judging two anonymous responses to a person seeking mental-health support.

Person's message:
<question>
{question}
</question>

Response A:
<response_a>
{a}
</response_a>

Response B:
<response_b>
{b}
</response_b>

Which response is better counseling: more empathetic, more specific to the person's situation, more helpful and appropriate? Ignore length; a concise good answer beats a rambling one. Reply with exactly one word: A, B, or TIE."""

with open(RESULTS) as f:
    data = json.load(f)

base_outs = data["models"]["unsloth/Llama-3.2-3B"]["all_outputs"]
ft_outs = data["models"]["seasalt29/model3"]["all_outputs"]
assert len(base_outs) == len(ft_outs)

# Recreate the same eval questions (same dataset, seed, sample size as the eval)
from datasets import load_dataset
ds = load_dataset("Amod/mental_health_counseling_conversations", split="train")
random.seed(data["seed"])
idx = random.sample(range(len(ds)), data["n_samples"])
questions = [ds[i]["Context"] for i in idx]

rng = random.Random(SEED)
records = []
wins = {"finetuned": 0, "base": 0, "tie": 0, "invalid": 0}

for n, (q, b_out, f_out) in enumerate(zip(questions, base_outs, ft_outs)):
    ft_is_a = rng.random() < 0.5
    a, b = (f_out, b_out) if ft_is_a else (b_out, f_out)
    prompt = JUDGE_PROMPT.format(question=q[:2000], a=a[:2000], b=b[:2000])
    body = {
        "model": JUDGE_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "reasoning_effort": "low",
        "max_completion_tokens": 2000,
    }
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
                 "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            out = json.load(resp)
        verdict = (out["choices"][0]["message"]["content"] or "").strip().upper()
    except Exception as e:
        verdict = f"ERROR:{e}"[:80]
    if verdict == "A":
        winner = "finetuned" if ft_is_a else "base"
    elif verdict == "B":
        winner = "base" if ft_is_a else "finetuned"
    elif verdict == "TIE":
        winner = "tie"
    else:
        winner = "invalid"
    wins[winner] += 1
    records.append({"i": n, "ft_is_a": ft_is_a, "verdict": verdict, "winner": winner})
    print(f"{n+1}/{len(questions)}: {winner}  (running: {wins})", flush=True)

decided = wins["finetuned"] + wins["base"]
summary = {
    "judge_model": JUDGE_MODEL,
    "n": len(questions),
    "wins": wins,
    "finetuned_win_rate_incl_ties": wins["finetuned"] / len(questions),
    "finetuned_win_rate_of_decided": wins["finetuned"] / decided if decided else None,
    "position_randomization_seed": SEED,
    "records": records,
}
with open(OUT, "w") as f:
    json.dump(summary, f, indent=2)
print(json.dumps({k: v for k, v in summary.items() if k != "records"}, indent=2))
