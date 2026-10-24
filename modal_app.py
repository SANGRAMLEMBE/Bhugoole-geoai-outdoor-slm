"""Deploy the API to Modal (serverless containers, scales to zero, stable https URL).

    pip install modal && modal token new      # once: sign in
    modal deploy modal_app.py                 # prints https://<workspace>--bhugoole-geoai-web.modal.run
    modal run modal_app.py::refresh_osm       # rebuild the OSM replica now (it also runs every day)

The fine-tuned GGUF is baked into the image at build time. Map features come from live Overpass first; when the
public Overpass servers are overloaded, from our own OpenStreetMap replica of India, which `refresh_osm` rebuilds
every day from Geofabrik's daily extract into a persistent volume. The copy in osm_fallback/ is used only until
the first rebuild has run.
"""
import os
import sys
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent
GGUF_REPO = "WIZARDIAN/Bhugoole-geoai-outdoor-slm"
GGUF_FILE = "bhugoole-geoai-outdoor-slm-q4_k_m.gguf"
GGUF_REVISION = "144a5f86359b0e7c959596b6a959c1afbd139dfc"  # version 2 of the fine-tune, see finetune/TRAINING_REPORT.md
CPU = 4
OSM_DIR = "/data/osm"
OSM_DB = f"{OSM_DIR}/osm_india.sqlite"
GEOFABRIK_INDIA = "https://download.geofabrik.de/asia/india-latest.osm.pbf"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("build-essential", "cmake", "libgomp1")
    .pip_install_from_requirements(str(ROOT / "requirements.txt"))
    # Prebuilt CPU wheel when available; a source build uses portable flags (no -march=native)
    .pip_install("llama-cpp-python>=0.3.2", extra_index_url="https://abetlen.github.io/llama-cpp-python/whl/cpu",
                 extra_options="--prefer-binary", env={"CMAKE_ARGS": "-DGGML_NATIVE=OFF"})
    .pip_install("osmium>=4.0")
    .run_commands(f"python -c \"from huggingface_hub import hf_hub_download as d; "
                  f"d('{GGUF_REPO}', '{GGUF_FILE}', revision='{GGUF_REVISION}', local_dir='/models')\"")
    .env({"SLM_BACKEND": "llamacpp", "SLM_GGUF_PATH": f"/models/{GGUF_FILE}", "SLM_GGUF_REPO": GGUF_REPO,
          "SLM_GGUF_FILE": GGUF_FILE, "SLM_THREADS": str(CPU), "PYTHONUNBUFFERED": "1"})
    .add_local_dir(ROOT / "app", "/root/app", ignore=["**/__pycache__/**", "**/*.pyc"])
    .add_local_dir(ROOT / "scripts", "/root/scripts", ignore=["**/__pycache__/**", "**/*.pyc"])
    .add_local_dir(ROOT / "osm_fallback", "/root/osm_fallback")
)

app = modal.App("bhugoole-geoai", image=image)
osm_volume = modal.Volume.from_name("bhugoole-osm", create_if_missing=True)


# The model writes one answer at a time per container, so spread simultaneous questions over more containers
@app.function(cpu=CPU, memory=4096, timeout=300, scaledown_window=600, max_containers=4,
              volumes={OSM_DIR: osm_volume})
@modal.concurrent(max_inputs=2)
@modal.asgi_app()
def web():
    if os.path.isfile(OSM_DB):  # the daily replica; otherwise the copy baked into the image
        os.environ["OSM_LOCAL_DB"] = OSM_DB
    from app.main import app as fastapi_app
    return fastapi_app


# Geofabrik publishes the India extract once a day; rebuild our replica a couple of hours later
@app.function(cpu=2, memory=16384, timeout=3600, volumes={OSM_DIR: osm_volume},
              schedule=modal.Cron("0 14 * * *"))
def refresh_osm():
    import shutil
    import subprocess
    import urllib.request

    pbf, built = "/tmp/india-latest.osm.pbf", "/tmp/osm_india.sqlite"
    req = urllib.request.Request(GEOFABRIK_INDIA, headers={"User-Agent": "Bhugoole-GeoAI-Outdoor-Assistant/1.0"})
    with urllib.request.urlopen(req, timeout=600) as r, open(pbf, "wb") as f:
        shutil.copyfileobj(r, f, length=8 << 20)
    print(f"downloaded {os.path.getsize(pbf) / 1e6:.0f} MB", flush=True)
    subprocess.run([sys.executable, "/root/scripts/build_osm_fallback.py", pbf, "--out", built], check=True)
    # Build on local disk, then swap the finished file into the volume in one step
    shutil.copyfile(built, OSM_DB + ".new")
    os.replace(OSM_DB + ".new", OSM_DB)
    osm_volume.commit()
    print(f"replica updated: {os.path.getsize(OSM_DB) / 1e6:.0f} MB", flush=True)
