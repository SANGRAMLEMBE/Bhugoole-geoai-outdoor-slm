"""The small language model (<= 3B parameters), run locally.

Two jobs only, both grounded:
  1. parse()   - turn a messy question into schema-constrained JSON slots.
  2. compose() - write the user-facing answer from a fact sheet built from live data.
The model never sees the internet and is never asked to recall weather or places.
"""
from __future__ import annotations

import json
import logging
import re
import threading
import time

from . import config
from .activities import ACTIVITIES
from .profiles import PROFILES

log = logging.getLogger("slm")

PARSE_SCHEMA = {
    "type": "object",
    "properties": {
        "activity": {"type": "string", "enum": [*ACTIVITIES.keys(), "out_of_scope", "unknown"]},
        "location": {"type": "string"},
        "radius_km": {"type": "number"},
        "profiles": {"type": "array", "items": {"type": "string", "enum": list(PROFILES.keys())}},
    },
    "required": ["activity", "location", "radius_km", "profiles"],
}

PARSE_SYSTEM = (
    "You extract structured fields from a question about an outdoor activity in India. "
    f"activity must be one of {list(ACTIVITIES.keys())}, or \"out_of_scope\" if the question is not about "
    "going outdoors, or \"unknown\" if unclear. location is the place name copied exactly from the question, or "
    "an empty string if none is written. radius_km is the distance the question states, or 0. profiles lists any of "
    f"{list(PROFILES.keys())} that the question mentions (senior = elderly or parents, child = kids). "
    "Reply with JSON only."
)
PARSE_SHOTS = [
    ("Powai morning walk tmrw? my mom is 72",
     {"activity": "walking", "location": "Powai", "radius_km": 0, "profiles": ["senior"]}),
    ("any decent place to shoot some pics near Colaba causeway",
     {"activity": "photography", "location": "Colaba causeway", "radius_km": 0, "profiles": []}),
    ("book me a train ticket to Delhi",
     {"activity": "out_of_scope", "location": "", "radius_km": 0, "profiles": []}),
]

COMPOSE_SYSTEM = (
    "You are a friendly outdoor-activity advisor for India. Write the answer using ONLY the facts in the "
    "FACT SHEET. Never add a number, place, time or weather detail that is not in the fact sheet, and copy "
    "place names and times exactly. Write 80-150 words of plain text (no headings, no tables): start with a "
    "one-sentence recommendation naming the place and time, then explain why in two or three short sentences "
    "using the strongest reasons, mention the main thing to avoid, and end with one practical tip. If the fact "
    "sheet says the activity is not advised, say so clearly first. Write flowing sentences: no labels such as "
    "\"Tip:\", and never copy the fact sheet's field names. Describe air quality only with the category the "
    "fact sheet gives (Good, Satisfactory, Moderate, Poor, Very Poor, Severe)."
)


class SLM:
    def __init__(self):
        self.backend = config.SLM_BACKEND
        self._model = None
        self._tok = None
        self._lock = threading.Lock()
        self._loading = threading.Lock()
        self.load_error: str | None = None
        self.load_seconds: float | None = None

    @property
    def available(self) -> bool:
        return self.backend != "none" and self._ensure() is not None

    @property
    def info(self) -> dict:
        return {"backend": self.backend, "model": config.SLM_DISPLAY_NAME, "hf_repo": config.HF_MODEL_REPO,
                "loaded": self._model is not None, "load_error": self.load_error, "load_seconds": self.load_seconds}

    def _ensure(self):
        if self._model is not None or self.backend == "none" or self.load_error:
            return self._model
        with self._loading:
            if self._model is not None or self.load_error:
                return self._model
            t0 = time.time()
            try:
                if self.backend == "llamacpp":
                    from llama_cpp import Llama
                    kw = dict(n_ctx=config.SLM_CTX, n_threads=config.SLM_THREADS, verbose=False)
                    if config.SLM_GGUF_PATH:
                        self._model = Llama(model_path=config.SLM_GGUF_PATH, **kw)
                    else:
                        self._model = Llama.from_pretrained(repo_id=config.SLM_GGUF_REPO,
                                                            filename=config.SLM_GGUF_FILE, **kw)
                elif self.backend == "transformers":
                    from transformers import AutoModelForCausalLM, AutoTokenizer
                    self._tok = AutoTokenizer.from_pretrained(config.SLM_HF_MODEL)
                    model = AutoModelForCausalLM.from_pretrained(config.SLM_HF_MODEL, torch_dtype="auto")
                    if config.SLM_ADAPTER:
                        from peft import PeftModel
                        model = PeftModel.from_pretrained(model, config.SLM_ADAPTER)
                    self._model = model.eval()
                else:
                    raise ValueError(f"unknown SLM_BACKEND {self.backend}")
                self.load_seconds = round(time.time() - t0, 1)
                log.info("SLM loaded in %.1fs", self.load_seconds)
            except Exception as exc:  # noqa: BLE001
                self.load_error = f"{type(exc).__name__}: {exc}"
                log.error("SLM failed to load, falling back to templates: %s", self.load_error)
        return self._model

    # ---- generation ----
    def _chat(self, messages: list[dict], max_tokens: int, json_schema: dict | None = None,
              temperature: float = 0.2) -> str:
        model = self._ensure()
        if model is None:
            raise RuntimeError("SLM unavailable")
        with self._lock:
            if self.backend == "llamacpp":
                kw = {}
                if json_schema is not None:
                    kw["response_format"] = {"type": "json_object", "schema": json_schema}
                out = model.create_chat_completion(messages=messages, max_tokens=max_tokens,
                                                   temperature=temperature, top_p=0.9, **kw)
                return out["choices"][0]["message"]["content"]
            import torch
            prompt = self._tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            ids = self._tok(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                gen = model.generate(**ids, max_new_tokens=max_tokens, do_sample=temperature > 0,
                                     temperature=max(temperature, 1e-3), top_p=0.9)
            return self._tok.decode(gen[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)

    def parse(self, question: str) -> dict:
        msgs = [{"role": "system", "content": PARSE_SYSTEM}]
        for q, a in PARSE_SHOTS:
            msgs += [{"role": "user", "content": q}, {"role": "assistant", "content": json.dumps(a)}]
        msgs.append({"role": "user", "content": question})
        raw = self._chat(msgs, 120, json_schema=PARSE_SCHEMA, temperature=0.0)
        m = re.search(r"\{.*\}", raw, re.S)
        return json.loads(m.group(0)) if m else {}

    def compose(self, fact_sheet: str, feedback: str | None = None) -> str:
        user = f"FACT SHEET\n{fact_sheet}\n\nWrite the answer now."
        if feedback:
            user += f"\n\nYour previous draft used details that are not in the fact sheet ({feedback}). Use only fact-sheet values."
        msgs = [{"role": "system", "content": COMPOSE_SYSTEM}, {"role": "user", "content": user}]
        return self._chat(msgs, config.SLM_MAX_TOKENS, temperature=0.3).strip()


SLM_MODEL = SLM()
