# AMD AI League - Mini-Challenge 3 (RAG). Saudi Sign Intelligence - grounded RAG on AMD Instinct.
# Base image: exactly the mandated one (checked by layer identity). It ships torch (ROCm) and PIL only.
FROM rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false \
    MODEL_DIR=/models/vlm MC3_INDEX_DIR=/app/index OUT_DIR=/app/output RAG_SOCK=/tmp/mc3_rag.sock

WORKDIR /app

# --no-deps everywhere: pip must never resolve torch (it would install a CUDA build over the ROCm one).
COPY requirements.lock requirements.base.txt /tmp/
RUN python3 -c "import torch; print('base torch', torch.__version__)" \
 && pip install --no-cache-dir --no-deps -r /tmp/requirements.base.txt \
 && pip install --no-cache-dir --no-deps -r /tmp/requirements.lock \
 && python3 -c "import torch, torchvision, transformers, tokenizers, pymupdf, docx, openpyxl, PIL; \
from transformers import AutoProcessor, AutoModelForImageTextToText; \
v = torch.__version__; print('torch', v, 'torchvision', torchvision.__version__, 'transformers', transformers.__version__); \
assert 'rocm' in v, 'torch is not the ROCm build'"

# Weights ship inside the image: there is no network at evaluation time.
COPY models/vlm /models/vlm

COPY app/ /app/
COPY requirements.lock /app/requirements.txt
RUN mkdir -p /app/corpus /app/output /app/index && chmod +x /app/start.sh

# Resident model server (loads the model once) + keep the container alive for the harness's docker exec calls.
CMD ["/app/start.sh"]
