"""QLoRA fine-tune of Qwen2.5-1.5B-Instruct for the parsing job.

Our run took 17 min on one RTX 4500 Ada (24 GB); a free Colab/Kaggle T4 also works, in ~30 min.

In Colab (Runtime -> T4 GPU):
    !git clone <your repo> && cd geoai-outdoor-assistant
    !pip -q install "transformers>=4.44" "peft>=0.12" "bitsandbytes>=0.43" accelerate datasets huggingface_hub
    !python finetune/make_parse_dataset.py --n 2500 --out finetune/data
    !python finetune/train_lora.py --data finetune/data --out finetune/out
    !huggingface-cli login
    !python scripts/push_model.py --repo <user>/geoai-outdoor-slm --adapter finetune/out/adapter --merged finetune/out/merged

The script evaluates the base model and the fine-tuned model on the held-out set (exact slot match)
and writes finetune/TRAINING_REPORT.md, which push_model.py puts into the model card.
"""
import argparse
import json
import math
import time
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments)

BASE = "Qwen/Qwen2.5-1.5B-Instruct"
REAL_EVAL = Path(__file__).parent / "real_eval.jsonl"  # hand-written questions, not from the generator


def real_eval_rows() -> list[dict]:
    """Realistic held-out questions in the same chat format as the synthetic data."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from make_parse_dataset import to_chat
    rows = []
    for line in REAL_EVAL.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            rows.append(to_chat(r["q"], {k: r[k] for k in ("activity", "location", "radius_km", "profiles")}))
    return rows


def encode(tok, ex, max_len):
    msgs = ex["messages"]
    prompt = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)
    target = msgs[-1]["content"] + tok.eos_token
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    t_ids = tok(target, add_special_tokens=False)["input_ids"]
    ids = (p_ids + t_ids)[:max_len]
    labels = ([-100] * len(p_ids) + t_ids)[:max_len]  # learn only the JSON answer
    return {"input_ids": ids, "labels": labels, "attention_mask": [1] * len(ids)}


def collate(tok):
    def fn(batch):
        n = max(len(b["input_ids"]) for b in batch)
        pad = tok.pad_token_id
        return {
            "input_ids": torch.tensor([b["input_ids"] + [pad] * (n - len(b["input_ids"])) for b in batch]),
            "labels": torch.tensor([b["labels"] + [-100] * (n - len(b["labels"])) for b in batch]),
            "attention_mask": torch.tensor([b["attention_mask"] + [0] * (n - len(b["attention_mask"])) for b in batch]),
        }
    return fn


@torch.no_grad()
def evaluate(model, tok, rows):
    model.eval()
    hits = {"exact": 0, "activity": 0, "location": 0, "profiles": 0, "valid_json": 0}
    for ex in rows:
        msgs = ex["messages"]
        gold = json.loads(msgs[-1]["content"])
        prompt = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)
        ids = tok(prompt, return_tensors="pt").to(model.device)
        out = model.generate(**ids, max_new_tokens=80, do_sample=False, pad_token_id=tok.pad_token_id)
        text = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
        try:
            pred = json.loads(text[text.index("{"): text.rindex("}") + 1])
            hits["valid_json"] += 1
        except ValueError:
            continue
        a = pred.get("activity") == gold["activity"]
        l = (pred.get("location") or "").lower() == gold["location"].lower()
        p = sorted(pred.get("profiles") or []) == sorted(gold["profiles"])
        hits["activity"] += a
        hits["location"] += l
        hits["profiles"] += p
        hits["exact"] += a and l and p
    return {k: round(v / len(rows), 3) for k, v in hits.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="finetune/data")
    ap.add_argument("--out", default="finetune/out")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--eval-n", type=int, default=120)
    a = ap.parse_args()
    out = Path(a.out)
    tok = AutoTokenizer.from_pretrained(BASE)
    tok.pad_token = tok.pad_token or tok.eos_token
    ds = load_dataset("json", data_files={"train": f"{a.data}/train.jsonl", "eval": f"{a.data}/eval.jsonl"})
    eval_rows = list(ds["eval"])[: a.eval_n]
    real_rows = real_eval_rows()

    fp16 = not torch.cuda.is_bf16_supported()
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                             bnb_4bit_compute_dtype=torch.float16 if fp16 else torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=bnb, device_map="auto")
    t0 = time.time()
    base_scores = evaluate(model, tok, eval_rows)
    base_real = evaluate(model, tok, real_rows)
    print("base:", base_scores, "real:", base_real)

    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                                             "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()
    train = ds["train"].map(lambda ex: encode(tok, ex, a.max_len), remove_columns=ds["train"].column_names)
    total_steps = math.ceil(len(train) / 16 * a.epochs)  # effective batch 4 x 4
    # warmup_steps (not warmup_ratio) works on transformers 4.x and 5.x
    args = TrainingArguments(output_dir=str(out / "ckpt"), num_train_epochs=a.epochs, per_device_train_batch_size=4,
                             gradient_accumulation_steps=4, learning_rate=a.lr, lr_scheduler_type="cosine",
                             warmup_steps=max(1, int(0.05 * total_steps)), logging_steps=10, save_strategy="no",
                             fp16=fp16, bf16=not fp16,
                             gradient_checkpointing=True, report_to=[])
    Trainer(model=model, args=args, train_dataset=train, data_collator=collate(tok)).train()
    model.save_pretrained(out / "adapter")
    tok.save_pretrained(out / "adapter")
    tuned_scores = evaluate(model, tok, eval_rows)
    tuned_real = evaluate(model, tok, real_rows)
    print("fine-tuned:", tuned_scores, "real:", tuned_real)

    # Merge into full-precision weights (for from_pretrained and for GGUF conversion)
    del model
    torch.cuda.empty_cache()
    full = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float16, device_map="cpu")
    merged = PeftModel.from_pretrained(full, out / "adapter").merge_and_unload()
    merged.save_pretrained(out / "merged", safe_serialization=True)
    tok.save_pretrained(out / "merged")

    n_train = len(ds["train"])
    report = f"""QLoRA fine-tune of `{BASE}` for slot extraction (question -> strict JSON).

* Data: {n_train} synthetic training questions and {len(eval_rows)} held-out questions, generated by
  `finetune/make_parse_dataset.py` (labels known by construction; typos, lower case, Hinglish fillers,
  missing places and out-of-scope requests are over-sampled).
* LoRA r=16, alpha=32, dropout 0.05 on all attention and MLP projections; 4-bit NF4 base;
  {a.epochs} epoch(s), learning rate {a.lr}, effective batch 16, loss on the JSON answer only.
* Hardware: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}; wall time {(time.time() - t0) / 60:.0f} min.

| Metric | Base, synthetic held-out ({len(eval_rows)}) | Fine-tuned, synthetic | Base, real questions ({len(real_rows)}) | Fine-tuned, real |
|---|---|---|---|---|
""" + "\n".join(f"| {k.replace('_', ' ')} | {base_scores[k]:.1%} | {tuned_scores[k]:.1%} | {base_real[k]:.1%} | "
                f"{tuned_real[k]:.1%} |" for k in base_scores) + """

The synthetic held-out set comes from the same templates as the training data, so it is optimistic.
The real-question set (`finetune/real_eval.jsonl`) is hand-written and never used for training.
"""
    (Path(__file__).parent / "TRAINING_REPORT.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
