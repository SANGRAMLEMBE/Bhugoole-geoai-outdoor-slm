"""Deploy the API to Google Cloud Run (stays inside the always-free tier at hackathon traffic).

    gcloud auth login
    python scripts/deploy_cloudrun.py --project <project-id> --billing <billing-account-id>

Builds the Dockerfile with Cloud Build (the fine-tuned GGUF is baked into the image, and the offline OSM
layer in osm_fallback/ is uploaded with the source; see .gcloudignore), then prints the endpoint URL.
Instances scale to zero when idle; the first request after a quiet spell takes ~20-40 s longer.
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def gcloud_cmd() -> list[str]:
    exe = shutil.which("gcloud") or shutil.which("gcloud.cmd")
    if not exe:
        local = Path(os.environ.get("LOCALAPPDATA", "")) / "gcloud-cli" / "google-cloud-sdk" / "bin" / "gcloud.cmd"
        exe = str(local) if local.is_file() else None
    if not exe:
        sys.exit("gcloud CLI not found: install it from https://cloud.google.com/sdk/docs/install")
    return [exe]


def run(*args: str, check: bool = True) -> str:
    cmd = gcloud_cmd() + list(args)
    print("$ gcloud " + " ".join(args), flush=True)
    res = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    if res.returncode != 0 and check:
        sys.exit(res.stderr.strip() or res.stdout.strip())
    return res.stdout.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--billing", help="billing account id to link (needed once per project)")
    ap.add_argument("--region", default="asia-south1")
    ap.add_argument("--service", default="bhugoole-geoai")
    ap.add_argument("--cpu", default="4")
    ap.add_argument("--memory", default="4Gi")
    a = ap.parse_args()
    if not (ROOT / "osm_fallback" / "osm_india.sqlite").is_file():
        print("WARNING: osm_fallback/osm_india.sqlite is missing (run scripts/build_osm_fallback.py); "
              "the endpoint will have no offline map layer when Overpass is down.")
    if a.billing:
        run("billing", "projects", "link", a.project, f"--billing-account={a.billing}")
    run("services", "enable", "run.googleapis.com", "cloudbuild.googleapis.com", "artifactregistry.googleapis.com",
        f"--project={a.project}")
    # Live progress for the build, which takes ~10 min
    subprocess.run(gcloud_cmd() + [
        "run", "deploy", a.service, "--source", ".", f"--project={a.project}", f"--region={a.region}",
        "--port=7860", f"--cpu={a.cpu}", f"--memory={a.memory}", "--timeout=300", "--concurrency=4",
        "--min-instances=0", "--max-instances=2", "--cpu-boost", "--allow-unauthenticated",
        f"--set-env-vars=SLM_THREADS={a.cpu}", "--quiet"], cwd=ROOT, check=True)
    url = run("run", "services", "describe", a.service, f"--project={a.project}", f"--region={a.region}",
              "--format=value(status.url)")
    print(f"\nEndpoint: {url}/ask\nHealth:   {url}/health\nDemo UI:  {url}/")


if __name__ == "__main__":
    main()
