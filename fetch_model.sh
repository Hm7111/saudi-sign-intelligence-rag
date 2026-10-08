#!/bin/sh
# Downloads the model weights that the Dockerfile copies into the image (no network at evaluation time).
set -e
python3 -m pip install -q "huggingface_hub>=1.0"
python3 -c "from huggingface_hub import snapshot_download; snapshot_download(\"Qwen/Qwen3-VL-8B-Instruct\", local_dir=\"models/vlm\")"
