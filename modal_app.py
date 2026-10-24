"""Deploy the API to Modal (serverless containers, scales to zero, stable https URL).

    pip install modal && modal token new      # once: sign in
    modal deploy modal_app.py                 # prints https://<workspace>--bhugoole-geoai-web.modal.run

The fine-tuned GGUF is baked into the image at build time; the app code and the offline OSM layer
(osm_fallback/osm_india.sqlite) are added from this checkout.
"""
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent
GGUF_REPO = "WIZARDIAN/Bhugoole-geoai-outdoor-slm"
GGUF_FILE = "bhugoole-geoai-outdoor-slm-q4_k_m.gguf"
CPU = 4

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("build-essential", "cmake", "libgomp1")
    .pip_install_from_requirements(str(ROOT / "requirements.txt"))
    # Prebuilt CPU wheel when available; a source build uses portable flags (no -march=native)
    .pip_install("llama-cpp-python>=0.3.2", extra_index_url="https://abetlen.github.io/llama-cpp-python/whl/cpu",
                 extra_options="--prefer-binary", env={"CMAKE_ARGS": "-DGGML_NATIVE=OFF"})
    .run_commands(f"python -c \"from huggingface_hub import hf_hub_download as d; "
                  f"d('{GGUF_REPO}', '{GGUF_FILE}', local_dir='/models')\"")
    .env({"SLM_BACKEND": "llamacpp", "SLM_GGUF_PATH": f"/models/{GGUF_FILE}", "SLM_GGUF_REPO": GGUF_REPO,
          "SLM_GGUF_FILE": GGUF_FILE, "SLM_THREADS": str(CPU), "PYTHONUNBUFFERED": "1"})
    .add_local_dir(ROOT / "app", "/root/app", ignore=["**/__pycache__/**", "**/*.pyc"])
    .add_local_dir(ROOT / "osm_fallback", "/root/osm_fallback")
)

app = modal.App("bhugoole-geoai", image=image)


@app.function(cpu=CPU, memory=4096, timeout=300, scaledown_window=600, max_containers=2)
@modal.concurrent(max_inputs=4)
@modal.asgi_app()
def web():
    from app.main import app as fastapi_app
    return fastapi_app
