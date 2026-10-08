# Saudi Sign Intelligence — Grounded RAG on AMD Instinct

A citation-exact **Retrieval-Augmented Generation** container for the AMD AI League (Mini-Challenge 3), built on the
mandated `rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0` image and an AMD Instinct GPU.

It answers a question about a folder of mixed documents with **the value only**, and cites **exactly** the files the
answer depends on — or returns an empty answer when the folder does not hold it.

```
python3 /app/app.py --index /app/corpus
python3 /app/app.py --corpus /app/corpus --query-id query_01 --query "What is the maximum junction temperature of the TQ-40?"
# -> /app/output/query_01_output.json  {"answer": "94", "citations": ["specs/tq40_datasheet_r2.pdf"], "confidence": 0.9}
```

## Architecture

```
container CMD ── app/start.sh ── app/server.py  (resident: Qwen3-VL-8B bf16 on the GPU, the index, a unix socket)
                                     ▲
harness ── python3 /app/app.py ──────┘  (standard library only; one short-lived process per question)
```

* **Resident model.** The model is loaded once by the container's CMD and warmed before it reports ready. `app.py` is a
  thin client, so every question process starts in milliseconds. A pid file guarantees a single resident copy, so the
  VRAM peak stays at one model (~20 GiB).
* **Defensive ingestion** (`app/ingest.py`). PDF (scanned pages are read by the vision model), DOCX paragraphs *and*
  tables in reading order, multi-sheet XLSX, CSV, logs/TXT/YAML, Python constants and keyword defaults (via `ast`),
  and images (OCR by the same VLM, with label→value pairing for diagrams). Empty folders, unknown types, unreadable
  files and encrypted PDFs are skipped per file — one bad file never stops the walk.
* **Superseded documents.** A file is *withdrawn* only when it says so about itself (file name or header status line),
  or when a newer revision of the same family exists. A current datasheet that mentions "revision 1 is withdrawn" in
  its history stays current. Withdrawn files are hidden from the model, so it never sees the decoy number.
* **Long-context first** (`app/rag.py`). When the corpus fits (~15k tokens), every current file goes into the prompt,
  ranked by BM25 + identifier hops; nothing can be "not retrieved". Long files (logs) are compressed to their events,
  warnings, errors and rare identifiers instead of being truncated, plus the passages that best match the question.
* **Citations verified in code.** The model names the file that *states* the value (with a verbatim quote) and, for
  chains, the link files that handed it an identifier. Code then checks that the value is really in the value file,
  re-reads image answers by showing the picture to the VLM, and keeps a link file only if it shares an identifier with
  the chain that the question itself does not contain (log → ticket → KB article).
* **Refusal.** No grounded value → `{"answer": "", "citations": [], "confidence": 0.0}`. An empty, valid JSON is written
  first, so a crash or timeout can never produce a malformed file.

## Results

| Set | Score | Notes |
|---|---|---|
| Official sample (10 questions) | **10 / 10** | in-process and through the real harness path (one process per question) |
| Self-built hard set (20 questions, `tools/make_hard_corpus.py`) | **20 / 20** | superseded revisions flagged only in their header, near-identical products, multi-sheet near-miss rows, scanned PDF, rotated label photo, pin-out diagram, 2- and 3-file chains, logs larger than the context budget, unanswerables |

Per question: 3–6 s (budget 30 s). Model load ~10 s, index ~10 s (budget 10 min).

## Build

```
./fetch_model.sh                          # Qwen/Qwen3-VL-8B-Instruct -> models/vlm (Apache-2.0)
docker build -t <you>/<repo>:<tag> .
docker run -d --network none --device=/dev/kfd --device=/dev/dri --group-add video \
           --cap-drop DAC_OVERRIDE <you>/<repo>:<tag>
```

Every `pip install` in the Dockerfile uses `--no-deps` with pinned versions (`requirements.lock`), and the build asserts
that `torch.__version__` is still the ROCm build. Image ≈ 46 GiB (base 29.2 GiB + weights 16.4 GiB), under the 60 GiB limit.

## Why

Built in Riyadh by a Saudi Sign Language interpreter and AI developer. The same grounded-answer engine — answer only
from the documents, always show the source — is the foundation for trustworthy document Q&A for the Deaf community.
