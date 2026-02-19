"""
LTX-2 LoRA Training — RunPod Pod Edition.

Starts training on boot, uploads checkpoints as it goes, terminates pod when done.
All data lives under /workspace (persistent volume).

Config via environment variables:
    DATASET         - Dataset name in S3 (required)
    CONFIG          - Config name in S3 without .yaml (required)
    PREPROCESS      - "true" to preprocess raw data (default: false)
    RESOLUTION      - Resolution bucket (default: 768x768x1)
    RESUME          - "true" to resume from latest S3 checkpoint (default: false)
    STEPS           - Override step count (optional)
    TERMINATE       - "true" to terminate pod when done (default: true)
    WITH_AUDIO      - "true" to preprocess audio latents (default: false)

    S3_ENDPOINT_URL, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY - S3 creds
    HF_TOKEN        - HuggingFace token for model downloads
    RUNPOD_API_KEY  - For self-termination
    RUNPOD_POD_ID   - Auto-set by RunPod
"""

import os
import subprocess
import sys
import time

# ============================================================================
# PATHS — everything under /workspace for persistence
# ============================================================================

WORKSPACE = "/workspace"
MODEL_DIR = f"{WORKSPACE}/models"
DATA_DIR = f"{WORKSPACE}/data"

# Support both Docker (/app/LTX-2) and bootstrap (/workspace/LTX-2) installs
LTX2_DIR = "/app/LTX-2" if os.path.exists("/app/LTX-2/.venv/bin/python") else f"{WORKSPACE}/LTX-2"
LTX2_PYTHON = f"{LTX2_DIR}/.venv/bin/python"

S3_BUCKET = "training-data"
S3_PREFIX = "ltx2"

LTX2_REPO = "Lightricks/LTX-2"
LTX2_MODEL_FILE = "ltx-2-19b-dev.safetensors"
GEMMA_REPO = "google/gemma-3-12b-it-qat-q4_0-unquantized"


# ============================================================================
# S3
# ============================================================================

def get_s3():
    import boto3
    return boto3.client(
        "s3",
        endpoint_url=os.environ["S3_ENDPOINT_URL"],
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    )


def sync_from_s3(s3_prefix, local_path):
    s3 = get_s3()
    os.makedirs(local_path, exist_ok=True)
    paginator = s3.get_paginator("list_objects_v2")
    prefix = s3_prefix.lstrip("/")
    count = 0
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            rel = key[len(prefix):].lstrip("/")
            if not rel:
                continue
            local_file = os.path.join(local_path, rel)
            os.makedirs(os.path.dirname(local_file), exist_ok=True)
            s3.download_file(S3_BUCKET, key, local_file)
            count += 1
    return count


def sync_to_s3(local_path, s3_prefix, verify_safetensors=False):
    s3 = get_s3()
    prefix = s3_prefix.lstrip("/")
    count = 0
    for root, dirs, files in os.walk(local_path):
        for f in files:
            fp = os.path.join(root, f)
            rel = os.path.relpath(fp, local_path)
            key = f"{prefix}/{rel}"
            size = os.path.getsize(fp)
            print(f"  [upload] {key} ({size / 1024 / 1024:.1f}MB)", flush=True)
            s3.upload_file(fp, S3_BUCKET, key)
            count += 1
            if verify_safetensors and f.endswith(".safetensors"):
                resp = s3.head_object(Bucket=S3_BUCKET, Key=key)
                if resp["ContentLength"] != size:
                    raise RuntimeError(
                        f"Size mismatch for {key}: local={size} remote={resp['ContentLength']}"
                    )
                print(f"    verified OK", flush=True)
    return count


def start_background_sync(local_path, s3_prefix, interval=60):
    import threading
    uploaded = {}
    stop = threading.Event()

    def loop():
        s3 = get_s3()
        while not stop.wait(interval):
            try:
                now = time.time()
                for root, dirs, files in os.walk(local_path):
                    for f in files:
                        fp = os.path.join(root, f)
                        try:
                            mtime = os.path.getmtime(fp)
                            size = os.path.getsize(fp)
                        except OSError:
                            continue
                        if (now - mtime) < 30 or size == 0:
                            continue
                        rel = os.path.relpath(fp, local_path)
                        key = f"{s3_prefix}/{rel}"
                        if key in uploaded and mtime <= uploaded[key]:
                            continue
                        s3.upload_file(fp, S3_BUCKET, key)
                        uploaded[key] = mtime
                        print(f"  [bg-sync] {key} ({size / 1024 / 1024:.1f}MB)", flush=True)
            except Exception as e:
                print(f"  [bg-sync] error: {e}", flush=True)

    t = threading.Thread(target=loop, daemon=True)
    t.start()
    return stop


# ============================================================================
# MODELS
# ============================================================================

def ensure_models():
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["HF_HOME"] = f"{WORKSPACE}/hf_cache"
    from huggingface_hub import hf_hub_download, snapshot_download

    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, LTX2_MODEL_FILE)
    te_path = os.path.join(MODEL_DIR, "gemma-3-12b-it-qat-q4_0-unquantized")

    if not os.path.exists(model_path):
        print(f"Downloading LTX-2 model (~43GB)...", flush=True)
        hf_hub_download(LTX2_REPO, LTX2_MODEL_FILE, local_dir=MODEL_DIR)
        print(f"  Done: {os.path.getsize(model_path) / 1024**3:.1f}GB", flush=True)
    else:
        print(f"LTX-2 model cached ({os.path.getsize(model_path) / 1024**3:.1f}GB)", flush=True)

    if not os.path.exists(te_path) or not os.listdir(te_path):
        print(f"Downloading Gemma text encoder...", flush=True)
        snapshot_download(GEMMA_REPO, local_dir=te_path)
        print("  Done", flush=True)
    else:
        print("Gemma text encoder cached", flush=True)

    return model_path, te_path


# ============================================================================
# CONFIG
# ============================================================================

def patch_config(config_path, dataset_local, output_dir, model_path, te_path,
                 resume_checkpoint=None, steps_override=None):
    import yaml
    with open(config_path) as f:
        config = yaml.safe_load(f)

    config["model"]["model_path"] = model_path
    config["model"]["text_encoder_path"] = te_path
    config["data"]["preprocessed_data_root"] = dataset_local
    config["output_dir"] = output_dir

    if resume_checkpoint:
        config["model"]["load_checkpoint"] = resume_checkpoint

    if steps_override:
        config["optimization"]["steps"] = int(steps_override)

    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"Config patched:", flush=True)
    print(f"  model: {model_path}", flush=True)
    print(f"  text_encoder: {te_path}", flush=True)
    print(f"  data: {dataset_local}", flush=True)
    print(f"  output: {output_dir}", flush=True)
    print(f"  steps: {config['optimization']['steps']}", flush=True)
    print(f"  lr: {config['optimization']['learning_rate']}", flush=True)
    if resume_checkpoint:
        print(f"  resume: {resume_checkpoint}", flush=True)
    return config


# ============================================================================
# POD TERMINATION
# ============================================================================

def terminate_pod():
    """Self-stop this RunPod pod (stops billing, keeps volume data)."""
    import urllib.request
    import json

    api_key = os.environ.get("RUNPOD_API_KEY")
    pod_id = os.environ.get("RUNPOD_POD_ID")
    if not api_key or not pod_id:
        print("Cannot self-terminate: missing RUNPOD_API_KEY or RUNPOD_POD_ID", flush=True)
        return

    def run_mutation(name, query):
        req = urllib.request.Request(
            "https://api.runpod.io/graphql",
            data=json.dumps({"query": query}).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req) as resp:
            result = resp.read().decode()
            print(f"  {name}: {result}", flush=True)
            return result

    print(f"Terminating pod {pod_id}...", flush=True)
    try:
        run_mutation("podTerminate", f'mutation {{ podTerminate(input: {{ podId: "{pod_id}" }}) }}')
    except Exception as e:
        print(f"  podTerminate failed: {e}", flush=True)
        print("  Please terminate the pod manually to avoid charges!", flush=True)


# ============================================================================
# MAIN
# ============================================================================

def main():
    dataset = os.environ.get("DATASET")
    config_name = os.environ.get("CONFIG")
    do_preprocess = os.environ.get("PREPROCESS", "false").lower() == "true"
    resolution = os.environ.get("RESOLUTION", "768x768x1")
    resume = os.environ.get("RESUME", "false").lower() == "true"
    steps_override = os.environ.get("STEPS")
    do_terminate = os.environ.get("TERMINATE", "true").lower() == "true"
    with_audio = os.environ.get("WITH_AUDIO", "false").lower() == "true"

    if not dataset or not config_name:
        print("ERROR: DATASET and CONFIG environment variables are required.", flush=True)
        print("  DATASET=bp_rework CONFIG=bp_rework_lora PREPROCESS=true", flush=True)
        sys.exit(1)

    print(f"\n{'='*60}", flush=True)
    print(f"LTX-2 LoRA Training", flush=True)
    print(f"  dataset:    {dataset}", flush=True)
    print(f"  config:     {config_name}", flush=True)
    print(f"  preprocess: {do_preprocess} ({resolution})", flush=True)
    print(f"  resume:     {resume}", flush=True)
    print(f"  steps:      {steps_override or '(from config)'}", flush=True)
    print(f"  terminate:  {do_terminate}", flush=True)
    print(f"  with_audio: {with_audio}", flush=True)
    print(f"{'='*60}\n", flush=True)

    # ---- Step 1: Models ----
    print("=== Step 1: Models ===", flush=True)
    model_path, te_path = ensure_models()

    # ---- Step 2: Dataset ----
    print("\n=== Step 2: Dataset ===", flush=True)
    dataset_local = f"{DATA_DIR}/datasets/{dataset}"
    os.makedirs(dataset_local, exist_ok=True)

    if do_preprocess:
        raw_local = f"{DATA_DIR}/raw/{dataset}"
        print(f"Downloading raw dataset: {dataset}", flush=True)
        count = sync_from_s3(f"{S3_PREFIX}/raw/{dataset}", raw_local)
        print(f"  {count} files", flush=True)
    else:
        print(f"Downloading preprocessed dataset: {dataset}", flush=True)
        count = sync_from_s3(f"{S3_PREFIX}/datasets/{dataset}", dataset_local)
        print(f"  {count} files", flush=True)

    # ---- Step 3: Preprocess ----
    if do_preprocess:
        print("\n=== Step 3: Preprocessing ===", flush=True)
        dataset_json = f"{DATA_DIR}/raw/{dataset}/dataset.json"
        cmd = [
            LTX2_PYTHON,
            f"{LTX2_DIR}/packages/ltx-trainer/scripts/process_dataset.py",
            dataset_json,
            "--resolution-buckets", resolution,
            "--model-path", model_path,
            "--text-encoder-path", te_path,
            "--batch-size", "1",
        ]
        if with_audio:
            cmd.append("--with-audio")
        print(f"Running: {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, cwd=LTX2_DIR, check=True)

        import shutil
        if os.path.exists(dataset_local):
            shutil.rmtree(dataset_local)
        shutil.copytree(f"{DATA_DIR}/raw/{dataset}", dataset_local)

        print("Uploading preprocessed data to S3...", flush=True)
        sync_to_s3(dataset_local, f"{S3_PREFIX}/datasets/{dataset}")

    # ---- Step 4: Config ----
    print("\n=== Step 4: Config ===", flush=True)
    config_local = f"{DATA_DIR}/config.yaml"
    os.makedirs(os.path.dirname(config_local), exist_ok=True)
    s3 = get_s3()
    config_key = f"{S3_PREFIX}/configs/{config_name}.yaml"
    print(f"Downloading: {config_key}", flush=True)
    s3.download_file(S3_BUCKET, config_key, config_local)

    output_dir = f"{DATA_DIR}/output/{dataset}"
    os.makedirs(output_dir, exist_ok=True)

    # Handle resume
    resume_checkpoint = None
    if resume:
        print("Downloading previous checkpoints for resume...", flush=True)
        sync_from_s3(f"{S3_PREFIX}/output/{dataset}", output_dir)
        # Find latest .safetensors checkpoint
        checkpoints_dir = os.path.join(output_dir, "checkpoints")
        if os.path.isdir(checkpoints_dir):
            safetensors = sorted([f for f in os.listdir(checkpoints_dir)
                                  if f.endswith(".safetensors")])
            if safetensors:
                resume_checkpoint = os.path.join(checkpoints_dir, safetensors[-1])
                print(f"  Resuming from: {resume_checkpoint}", flush=True)
        if not resume_checkpoint:
            print("  No checkpoint found, starting fresh", flush=True)

    patch_config(config_local, dataset_local, output_dir, model_path, te_path,
                 resume_checkpoint, steps_override)

    # ---- Step 5: Patches ----
    print("\n=== Step 5: Patches ===", flush=True)
    patches = {
        "config.py": f"{LTX2_DIR}/packages/ltx-trainer/src/ltx_trainer/config.py",
        "trainer.py": f"{LTX2_DIR}/packages/ltx-trainer/src/ltx_trainer/trainer.py",
    }
    for filename, dest in patches.items():
        try:
            s3.download_file(S3_BUCKET, f"{S3_PREFIX}/patches/{filename}", dest)
            print(f"  Patched {filename}", flush=True)
        except Exception:
            print(f"  No patch for {filename}", flush=True)

    # ---- Step 6: Train ----
    print(f"\n{'='*60}", flush=True)
    print("Starting training", flush=True)
    print(f"{'='*60}\n", flush=True)

    output_s3_prefix = f"{S3_PREFIX}/output/{dataset}"
    stop_sync = start_background_sync(output_dir, output_s3_prefix, interval=60)

    try:
        result = subprocess.run(
            [LTX2_PYTHON, f"{LTX2_DIR}/packages/ltx-trainer/scripts/train.py",
             config_local, "--disable-progress-bars"],
            cwd=LTX2_DIR,
            capture_output=True,
            text=True,
        )
        # Print all output so it's visible in RunPod logs
        if result.stdout:
            print(result.stdout, flush=True)
        if result.stderr:
            print(result.stderr, file=sys.stderr, flush=True)
        if result.returncode != 0:
            print(f"\n{'='*60}", flush=True)
            print(f"TRAINING FAILED (exit code {result.returncode})", flush=True)
            # Print last 50 lines of stderr for quick diagnosis
            if result.stderr:
                lines = result.stderr.strip().split("\n")
                print("Last stderr lines:", flush=True)
                for line in lines[-50:]:
                    print(f"  {line}", flush=True)
            print(f"{'='*60}\n", flush=True)
            sys.exit(result.returncode)
    finally:
        stop_sync.set()
        time.sleep(5)  # let background sync finish any in-progress uploads

    # ---- Step 7: Final upload ----
    print(f"\n{'='*60}", flush=True)
    print("Final upload to S3", flush=True)
    print(f"{'='*60}\n", flush=True)

    count = sync_to_s3(output_dir, output_s3_prefix, verify_safetensors=True)
    print(f"Uploaded {count} files", flush=True)

    print(f"\n{'='*60}", flush=True)
    print("Training complete!", flush=True)
    print(f"Output: s3://{S3_BUCKET}/{output_s3_prefix}/", flush=True)
    print(f"{'='*60}\n", flush=True)

    # ---- Terminate ----
    if do_terminate:
        terminate_pod()
        # Sleep so the process doesn't exit and trigger a container restart
        # before the API call takes effect
        print("Waiting for termination...", flush=True)
        time.sleep(300)


if __name__ == "__main__":
    main()
