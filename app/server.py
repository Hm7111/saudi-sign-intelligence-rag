"""Resident daemon, started by the container CMD: loads the vision-language model ONCE onto the GPU, holds the index,
and serves the thin client /app/app.py over a unix socket (one JSON line per request).

ops: ping -> {ready, error} | index {corpus} -> {ok, files} | query {corpus, query, budget} -> {answer, citations, confidence}
"""
import os, sys, json, socket, threading, time, logging, traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ingest, rag

SOCK = os.environ.get("RAG_SOCK", "/tmp/mc3_rag.sock")
MODEL_DIR = os.environ.get("MODEL_DIR", "/models/vlm")
INDEX_DIR = os.environ.get("MC3_INDEX_DIR", "/app/index")
OCR_BATCH = int(os.environ.get("OCR_BATCH", "6"))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("server")

OCR_PROMPT = ("Transcribe ALL text visible in this image exactly as printed (keep codes, part numbers, revisions, "
              "units and symbols such as # exactly). If it is a diagram, table or pinout, write one line per item "
              "pairing each label with the label/value it is connected to or aligned with, e.g. 'B14 -> THERM_ALERT#'. "
              "Finally add one line 'DESCRIPTION: <what the image shows>'. Plain text only.")


class VLM:
    """Qwen3-VL in bf16 on the GPU (ROCm exposes it as 'cuda')."""

    def __init__(self, path):
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText
        self.torch = torch
        if not torch.cuda.is_available():
            raise RuntimeError("no GPU visible to torch")
        self.proc = AutoProcessor.from_pretrained(path)
        self.proc.tokenizer.padding_side = "left"
        kw = dict(dtype=torch.bfloat16, attn_implementation="sdpa")
        mm = os.environ.get("MAX_GPU_MEM")  # local testing on a small card only
        if mm:
            kw.update(device_map="auto", max_memory={0: mm, "cpu": "48GiB"})
        else:
            kw.update(device_map="cuda")
        self.model = AutoModelForImageTextToText.from_pretrained(path, **kw).eval()
        self.dev = self.model.device if not mm else "cuda"
        self.lock = threading.Lock()
        log.info("model loaded on %s (%s)", torch.cuda.get_device_name(0), torch.__version__)

    def _gen(self, messages_list, max_new_tokens):
        inputs = self.proc.apply_chat_template(messages_list, add_generation_prompt=True, tokenize=True, padding=True,
                                               return_dict=True, return_tensors="pt").to(self.dev)
        with self.lock, self.torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                      temperature=None, top_p=None, top_k=None)
        return self.proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    def generate(self, prompt, max_new_tokens=256):
        return self._gen([[{"role": "user", "content": [{"type": "text", "text": prompt}]}]], max_new_tokens)[0]

    @staticmethod
    def _img(path_or_bytes):
        from PIL import Image
        import io
        img = Image.open(io.BytesIO(path_or_bytes) if isinstance(path_or_bytes, bytes) else path_or_bytes)
        img.load()
        img = img.convert("RGB")
        if max(img.size) > 1600:
            img.thumbnail((1600, 1600))
        return img

    def look(self, image, prompt, max_new_tokens=64):
        msg = [{"role": "user", "content": [{"type": "image", "image": self._img(image)}, {"type": "text", "text": prompt}]}]
        return self._gen([msg], max_new_tokens)[0]

    def ocr_many(self, images):
        out = []
        for i in range(0, len(images), OCR_BATCH):
            batch = [[{"role": "user", "content": [{"type": "image", "image": self._img(p)},
                                                   {"type": "text", "text": OCR_PROMPT}]}] for p in images[i:i + OCR_BATCH]]
            try:
                out += self._gen(batch, 600)
            except Exception:  # a bad batch: retry one by one so one image cannot cost the others
                log.warning("ocr batch failed, retrying singly:\n%s", traceback.format_exc())
                for msg in batch:
                    try:
                        out += self._gen([msg], 600)
                    except Exception:
                        out.append("")
        return out


STATE = {"vlm": None, "index": None, "ready": False, "error": None, "root": None}


def do_index(root):
    t0 = time.time()
    files = ingest.walk_corpus(root)
    docs, images = [], []
    for ap, rp in files:
        d = ingest.parse_file(ap, rp, ocr_image=lambda p: "\0IMG", ocr_pdf_page=lambda b: STATE["vlm"].look(b, OCR_PROMPT, 600))
        if d:
            d["abs_path"] = ap
            docs.append(d)
            if d["text"] == "\0IMG":
                images.append(d)
    if images:
        texts = STATE["vlm"].ocr_many([d["abs_path"] for d in images])
        for d, t in zip(images, texts):
            d["text"] = (t or "").strip()
    docs = [d for d in docs if d["text"]]
    for d in docs:
        d["status"] = "withdrawn" if ingest.self_withdrawn(d["path"], d["text"]) else d["status"]
    ingest.mark_superseded(docs)
    idx = rag.Index(docs)
    os.makedirs(INDEX_DIR, exist_ok=True)
    idx.save(os.path.join(INDEX_DIR, "index.pkl"))
    STATE["index"], STATE["root"] = idx, root
    log.info("indexed %d/%d files (%d images) in %.1fs", len(docs), len(files), len(images), time.time() - t0)
    for d in docs:
        log.info("  %-9s %-5s %s", d["status"] + ("*" if d.get("hidden") else ""), d["kind"], d["path"])
    return {"ok": True, "files": len(docs), "seen": len(files), "images": len(images), "seconds": round(time.time() - t0, 1)}


def handle(req):
    op = req.get("op")
    if op == "ping":
        return {"ready": STATE["ready"], "error": STATE["error"]}
    if not STATE["ready"]:
        return {"error": "model not ready: %s" % STATE["error"]}
    if op == "index":
        return do_index(req["corpus"])
    if op == "query":
        deadline = time.time() + float(req.get("budget", 24))
        if STATE["index"] is None:
            p = os.path.join(INDEX_DIR, "index.pkl")
            if os.path.exists(p):
                STATE["index"] = rag.Index.load(p)
            else:
                do_index(req["corpus"])
        idx = STATE["index"]
        root = req.get("corpus") or STATE["root"]
        for d in idx.docs:  # images are re-read at query time from the corpus actually given
            d["abs_path"] = os.path.join(root, d["path"])
        res = rag.answer(idx, req["query"], STATE["vlm"], deadline)
        log.info("Q %r -> %s", req["query"], res)
        return res
    return {"error": "bad op"}


def serve_conn(conn):
    try:
        data = b""
        while not data.endswith(b"\n"):
            chunk = conn.recv(1 << 16)
            if not chunk:
                break
            data += chunk
        try:
            resp = handle(json.loads(data))
        except Exception as e:
            log.error(traceback.format_exc())
            resp = {"error": repr(e)}
        conn.sendall((json.dumps(resp, ensure_ascii=False) + "\n").encode())
    except Exception:
        log.error(traceback.format_exc())
    finally:
        conn.close()


def load_model():
    try:
        STATE["vlm"] = VLM(MODEL_DIR)
        STATE["vlm"].generate("Reply with OK.", 4)  # compile/warm the GPU kernels before reporting ready
        STATE["ready"] = True
        log.info("ready")
    except Exception as e:
        STATE["error"] = repr(e)
        log.error(traceback.format_exc())


PID_FILE = "/tmp/mc3_server.pid"


def main():
    if os.path.exists(SOCK):  # a live server already owns the socket: never load a second copy of the model
        try:
            c = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM); c.settimeout(2); c.connect(SOCK); c.close()
            log.info("another server is already listening on %s - exiting", SOCK)
            return
        except Exception:
            os.remove(SOCK)
    with open(PID_FILE, "w") as f:
        f.write(str(os.getpid()))
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(SOCK)
    srv.listen(32)
    threading.Thread(target=load_model, daemon=True).start()  # answer pings while the weights load
    log.info("listening on %s", SOCK)
    while True:
        conn, _ = srv.accept()
        threading.Thread(target=serve_conn, args=(conn,), daemon=True).start()


if __name__ == "__main__":
    main()
