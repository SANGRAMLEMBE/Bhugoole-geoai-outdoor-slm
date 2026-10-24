"""Create the public Hugging Face model repository required by the hackathon.

Mode A (base model used unchanged; the default):
    python scripts/push_model.py --repo <user>/geoai-outdoor-slm --mirror-weights
  Uploads the model card, our configuration (prompts, JSON schema, generation settings) and, with
  --mirror-weights, the unchanged Apache-2.0 base weights so `from_pretrained("<repo>")` works.
  Tip: run this from Google Colab or Kaggle; uploading ~3 GB is much faster there.

Mode B (after LoRA fine-tuning with finetune/train_lora.py):
    python scripts/push_model.py --repo <user>/geoai-outdoor-slm --adapter finetune/out/adapter --merged finetune/out/merged \
        --gguf finetune/out/gguf/geoai-outdoor-slm-q4_k_m.gguf
  Uploads the adapter (in the repo root, loadable with PEFT), the merged full model under merged/, and the
  Q4_K_M GGUF that the API runs (deploy with --gguf-repo <user>/geoai-outdoor-slm --gguf-file <its name>).
"""
import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import os  # noqa: E402

os.environ.setdefault("SLM_BACKEND", "none")
from app import config  # noqa: E402
from app.slm import COMPOSE_SYSTEM, PARSE_SCHEMA, PARSE_SHOTS, PARSE_SYSTEM  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--base", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--mirror-weights", action="store_true")
    ap.add_argument("--adapter")
    ap.add_argument("--merged")
    ap.add_argument("--gguf", help="quantised GGUF of the merged model, used by the llama.cpp runtime")
    ap.add_argument("--code-url", default="https://github.com/<you>/geoai-outdoor-assistant")
    ap.add_argument("--api-url", default="https://<user>-geoai-outdoor-assistant.hf.space/ask")
    a = ap.parse_args()
    api = HfApi()
    api.create_repo(a.repo, repo_type="model", exist_ok=True, private=False)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        if a.adapter:
            # Merged full weights in the root so from_pretrained("<repo>") loads the fine-tuned model directly
            if a.merged:
                shutil.copytree(a.merged, out, dirs_exist_ok=True)
                shutil.copytree(a.adapter, out / "adapter", dirs_exist_ok=True)
                sub = "adapter"
            else:
                shutil.copytree(a.adapter, out, dirs_exist_ok=True)
                sub = None
            if a.gguf:
                shutil.copy(a.gguf, out / Path(a.gguf).name)
            where = ("merged weights in the repo root, the LoRA adapter in `adapter/`" if sub
                     else "LoRA adapter in the repo root")
            changes = ("QLoRA fine-tune for extracting question slots as strict JSON (see Training); " + where
                       + (f", and the Q4_K_M GGUF the live API runs (`{Path(a.gguf).name}`)." if a.gguf else "."))
            variant = "LoRA fine-tuned"
            training = (ROOT / "finetune" / "TRAINING_REPORT.md").read_text(encoding="utf-8") if (ROOT / "finetune" / "TRAINING_REPORT.md").exists() \
                else "QLoRA (r=16, alpha=32) on synthetic question->JSON pairs; see finetune/ in the code repository."
            adapter_usage = ("\nThe LoRA adapter alone, on top of the base model:\n```python\nfrom peft import PeftModel\n"
                             f"base = AutoModelForCausalLM.from_pretrained(\"{a.base}\")\n"
                             f"model = PeftModel.from_pretrained(base, \"{a.repo}\""
                             + (f", subfolder=\"{sub}\"" if sub else "") + ")\n```\n")
            if a.gguf:
                adapter_usage += (f"\nWith llama.cpp (what the API uses):\n```python\nfrom llama_cpp import Llama\n"
                                  f"llm = Llama.from_pretrained(\"{a.repo}\", \"{Path(a.gguf).name}\")\n```\n")
        else:
            src = Path(snapshot_download(a.base, allow_patterns=["*.json", "*.txt", "*.model", "merges.txt"] +
                                         (["*.safetensors"] if a.mirror_weights else [])))
            for f in src.iterdir():
                if f.name.lower() not in ("readme.md", "license"):
                    shutil.copy(f, out / f.name)
            variant = "base model, unchanged"
            changes = ("None to the weights (an unchanged mirror of the base model). Our contribution is the "
                       "prompting, JSON-schema-constrained decoding and the grounding verifier around it.")
            training = "No training: the base model is used as released."
            adapter_usage = ""
        (out / "geoai_config.json").write_text(json.dumps({
            "base_model": a.base,
            "runtime_gguf": ({"repo": a.repo, "file": Path(a.gguf).name} if a.gguf
                             else {"repo": config.SLM_GGUF_REPO, "file": config.SLM_GGUF_FILE}),
            "parse": {"system": PARSE_SYSTEM, "few_shot": PARSE_SHOTS, "json_schema": PARSE_SCHEMA, "temperature": 0.0,
                      "max_tokens": 120},
            "compose": {"system": COMPOSE_SYSTEM, "temperature": 0.3, "top_p": 0.9, "max_tokens": config.SLM_MAX_TOKENS},
        }, indent=2))
        card = (ROOT / "hf" / "MODEL_CARD.md").read_text(encoding="utf-8")
        for k, v in {"VARIANT": variant, "CHANGES": changes, "CODE_URL": a.code_url, "API_URL": a.api_url,
                     "PARSE_SYSTEM": PARSE_SYSTEM, "COMPOSE_SYSTEM": COMPOSE_SYSTEM, "REPO": a.repo,
                     "ADAPTER_USAGE": adapter_usage, "TRAINING": training}.items():
            card = card.replace("{{" + k + "}}", v)
        (out / "README.md").write_text(card, encoding="utf-8")
        (out / "LICENSE").write_text("Apache License 2.0 - https://www.apache.org/licenses/LICENSE-2.0\n"
                                     f"Base model {a.base} is (c) Alibaba Cloud, licensed under Apache-2.0.\n")
        api.upload_folder(repo_id=a.repo, folder_path=str(out), commit_message="Upload GeoAI Outdoor SLM")
    print(f"Done: https://huggingface.co/{a.repo}")


if __name__ == "__main__":
    main()
