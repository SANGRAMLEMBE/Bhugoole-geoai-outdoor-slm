---
license: apache-2.0
base_model: Qwen/Qwen2.5-1.5B-Instruct
language:
- en
pipeline_tag: text-generation
library_name: transformers
tags:
- geoai
- geospatial
- outdoor-recommendation
- grounded-generation
- small-language-model
- hackathon
---

# Bhugoole GeoAI Outdoor SLM ({{VARIANT}})

The language component of **GeoAI Outdoor Assistant**, built for the GeoAI Hackathon 2026
(Centre of Studies in Resources Engineering, IIT Bombay). The assistant answers questions such as
*"I have mild asthma, where's good for a morning walk near Powai in the next two days?"* by fetching
live OpenStreetMap, weather and air-quality data, scoring every nearby place hour by hour, and using
this small model to understand the question and explain the result.

| | |
|---|---|
| Base model | [`Qwen/Qwen2.5-1.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) (1.54B parameters, Apache-2.0) |
| Changes | {{CHANGES}} |
| Runtime | 4-bit GGUF (Q4_K_M) through llama.cpp on 2 CPU cores; `transformers` also supported |
| Licence | Apache-2.0 (same as the base model) |
| Code | {{CODE_URL}} |
| Live API | {{API_URL}} |

## What the model does, and what it never does

The model has two narrow jobs:

1. **Parse** a messy question into strict JSON (activity, place as written, radius, personal profile),
   with output constrained by a JSON-schema grammar. Used only when rule-based parsing leaves a gap.
   Any place the model returns must appear verbatim in the question, or it is discarded.
2. **Compose** a 80–150 word answer from a *fact sheet* that the system builds from data fetched live
   for that question (Nominatim, Overpass with an offline Geofabrik copy as fallback, Open-Meteo forecast
   and air quality, OSRM).

The model is never asked to recall weather, air quality or places. Every number, time and proper noun
in its answer is checked against the fact sheet by a grounding verifier; a draft that fails is
regenerated once with feedback, and if it fails again the system falls back to a deterministic template.

## Prompts

Parse (system prompt, few-shot, temperature 0, JSON schema grammar):
```
{{PARSE_SYSTEM}}
```

Compose (system prompt, temperature 0.3, max 320 tokens):
```
{{COMPOSE_SYSTEM}}
```

The exact JSON schema and generation settings are in `geoai_config.json` in this repository.

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
repo = "{{REPO}}"
tok = AutoTokenizer.from_pretrained(repo)
model = AutoModelForCausalLM.from_pretrained(repo, torch_dtype="auto")
```
{{ADAPTER_USAGE}}

## Training

{{TRAINING}}

## Intended use and limitations

* Intended for explaining outdoor-activity recommendations in India from supplied data. It is not a
  weather or medical authority; health advice is limited to general precautions.
* English only. Hinglish words in questions are handled mostly by the rule-based parser.
* A 1.5B model can still phrase things awkwardly or omit details; the verifier blocks invented facts but
  cannot guarantee fluency.

## Disclosure

The running assistant uses only this open model; no hosted language model is called.
