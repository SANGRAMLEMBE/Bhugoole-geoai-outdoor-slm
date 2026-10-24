# Hugging Face Spaces (Docker SDK, free CPU tier: 2 vCPU / 16 GB RAM) or any Docker host.
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends build-essential cmake libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Spaces run containers as uid 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH HF_HOME=/home/user/.cache/huggingface \
    PYTHONUNBUFFERED=1 SLM_BACKEND=llamacpp
WORKDIR $HOME/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt
# Prebuilt CPU wheel when available; otherwise builds from source (portable flags, ~5-10 min)
RUN CMAKE_ARGS="-DGGML_NATIVE=OFF" pip install --no-cache-dir --user "llama-cpp-python>=0.3.2" \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu

# Bake the model into the image so cold starts don't download it.
ARG SLM_GGUF_REPO=WIZARDIAN/Bhugoole-geoai-outdoor-slm
ARG SLM_GGUF_FILE=bhugoole-geoai-outdoor-slm-q4_k_m.gguf
ENV SLM_GGUF_REPO=$SLM_GGUF_REPO SLM_GGUF_FILE=$SLM_GGUF_FILE
RUN python -c "from huggingface_hub import hf_hub_download as d; import os; d(os.environ['SLM_GGUF_REPO'], os.environ['SLM_GGUF_FILE'])"

COPY --chown=user app ./app
# Offline OSM layer (Geofabrik extract), used only when the live Overpass API fails
COPY --chown=user osm_fallback ./osm_fallback
EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
