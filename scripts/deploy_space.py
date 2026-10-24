"""Deploy the API to a Hugging Face Docker Space (Docker Spaces need a paid plan since Oct 2026; see modal_app.py).

    pip install huggingface_hub
    huggingface-cli login            # a token with write access
    python scripts/deploy_space.py --space <user>/geoai-outdoor-assistant --model-repo <user>/geoai-outdoor-slm

The endpoint will be  https://<user>-geoai-outdoor-assistant.hf.space/ask
"""
import argparse
import io
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]

SPACE_README = """---
title: GeoAI Outdoor Assistant
emoji: 🧭
colorFrom: green
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
license: apache-2.0
short_description: Where and when to head outdoors, grounded in live open data
---

API for the GeoAI Hackathon 2026 (CSRE, IIT Bombay).

* `POST /ask` with `{{"question": "Where can I go running tomorrow evening near Koregaon Park?"}}`
* `GET /ask?q=...`, `GET /health`, `GET /docs`, demo UI at `/`

SLM: `{model}` (see the model card at https://huggingface.co/{model_repo}).
Data: OpenStreetMap (Nominatim, Overpass), Open-Meteo forecast and air quality (CAMS), OSRM. © OpenStreetMap contributors.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", required=True, help="<user>/<space-name>")
    ap.add_argument("--model-repo", required=True, help="your public HF model repo, e.g. <user>/geoai-outdoor-slm")
    ap.add_argument("--gguf-repo", default="Qwen/Qwen2.5-1.5B-Instruct-GGUF")
    ap.add_argument("--gguf-file", default="qwen2.5-1.5b-instruct-q4_k_m.gguf")
    ap.add_argument("--display-name", default="Qwen2.5-1.5B-Instruct (Q4_K_M GGUF via llama.cpp)")
    a = ap.parse_args()
    api = HfApi()
    api.create_repo(a.space, repo_type="space", space_sdk="docker", exist_ok=True)
    for k, v in {"HF_MODEL_REPO": a.model_repo, "SLM_GGUF_REPO": a.gguf_repo, "SLM_GGUF_FILE": a.gguf_file,
                 "SLM_DISPLAY_NAME": a.display_name}.items():
        api.add_space_variable(a.space, k, v)
    readme = SPACE_README.format(model=a.display_name, model_repo=a.model_repo)
    # Dockerfile build args follow the variables above
    dockerfile = (ROOT / "Dockerfile").read_text()
    dockerfile = dockerfile.replace("ARG SLM_GGUF_REPO=Qwen/Qwen2.5-1.5B-Instruct-GGUF", f"ARG SLM_GGUF_REPO={a.gguf_repo}")
    dockerfile = dockerfile.replace("ARG SLM_GGUF_FILE=qwen2.5-1.5b-instruct-q4_k_m.gguf", f"ARG SLM_GGUF_FILE={a.gguf_file}")
    if not (ROOT / "osm_fallback" / "osm_india.sqlite").is_file():
        print("WARNING: osm_fallback/osm_india.sqlite is missing (run scripts/build_osm_fallback.py); "
              "the Space will have no offline map layer when Overpass is down.")
    api.upload_folder(repo_id=a.space, repo_type="space", folder_path=str(ROOT),
                      allow_patterns=["app/**", "requirements.txt", "osm_fallback/**"],
                      ignore_patterns=["**/__pycache__/**"], commit_message="Deploy app")
    api.upload_file(path_or_fileobj=io.BytesIO(dockerfile.encode()), path_in_repo="Dockerfile", repo_id=a.space,
                    repo_type="space", commit_message="Dockerfile")
    api.upload_file(path_or_fileobj=io.BytesIO(readme.encode()), path_in_repo="README.md", repo_id=a.space,
                    repo_type="space", commit_message="Space card")
    user, name = a.space.split("/")
    print(f"Deployed. Build logs: https://huggingface.co/spaces/{a.space}")
    print(f"Endpoint (after the build finishes): https://{user}-{name}.hf.space/ask".replace("_", "-").lower())


if __name__ == "__main__":
    main()
