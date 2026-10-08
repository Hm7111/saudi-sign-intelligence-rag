#!/usr/bin/env python3
"""Thin client the harness invokes (standard library only, starts in milliseconds).

    python3 /app/app.py --index /app/corpus
    python3 /app/app.py --corpus /app/corpus --query-id query_01 --query "..."

The model and the index live in the resident server (server.py, started by the container CMD). A query always leaves a
valid /app/output/<query-id>_output.json behind: an empty refusal is written FIRST and replaced by the real answer,
so a crash or a timeout can never produce a malformed file.
"""
import os, sys, json, socket, time, argparse, subprocess

SOCK = os.environ.get("RAG_SOCK", "/tmp/mc3_rag.sock")
OUT_DIR = os.environ.get("OUT_DIR", "/app/output")
HERE = os.path.dirname(os.path.abspath(__file__))
EMPTY = {"answer": "", "citations": [], "confidence": 0.0}
QUERY_BUDGET = 30.0


def call(req, timeout):
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        s.connect(SOCK)
        s.sendall((json.dumps(req) + "\n").encode())
        data = b""
        while not data.endswith(b"\n"):
            chunk = s.recv(1 << 16)
            if not chunk:
                break
            data += chunk
        return json.loads(data)
    finally:
        s.close()


PID_FILE = "/tmp/mc3_server.pid"


def _alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except Exception:
        return False


def ensure_server():
    """The CMD starts the server. Only if NO server process exists at all (container started some other way) start one
    - never a second one: two resident copies of the model would double the VRAM peak."""
    t0 = time.time()
    while time.time() - t0 < 10:  # a just-started server needs a moment to write its pid and create its socket
        try:
            call({"op": "ping"}, 3)
            return
        except Exception:
            pass
        try:
            starting = _alive(int(open(PID_FILE).read().strip()))
        except Exception:
            starting = time.time() - t0 < 3  # no pid file yet: grace period before concluding there is no server
        if not starting:
            break
        time.sleep(0.25)
    try:
        if _alive(int(open(PID_FILE).read().strip())):
            return
    except Exception:
        pass
    subprocess.Popen([sys.executable, os.path.join(HERE, "server.py")], stdout=open("/tmp/mc3_server.log", "ab"),
                     stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)


def wait_ready(deadline):
    while time.time() < deadline:
        try:
            r = call({"op": "ping"}, 5)
            if r.get("ready"):
                return True
            if r.get("error"):
                return False
        except Exception:
            pass
        time.sleep(0.5)
    return False


def write(qid, obj):
    os.makedirs(OUT_DIR, exist_ok=True)
    tmp = os.path.join(OUT_DIR, f".{qid}.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
    os.replace(tmp, os.path.join(OUT_DIR, f"{qid}_output.json"))  # atomic: never a half-written file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index")
    ap.add_argument("--corpus", default="/app/corpus")
    ap.add_argument("--query-id")
    ap.add_argument("--query")
    a = ap.parse_args()
    t0 = time.time()

    if a.index:
        ensure_server()
        if wait_ready(t0 + 420):
            try:
                print(json.dumps(call({"op": "index", "corpus": os.path.abspath(a.index)}, 600 - (time.time() - t0) - 5)))
            except Exception as e:
                print(json.dumps({"error": repr(e)}))
        else:
            print(json.dumps({"error": "model server not ready"}))
        return 0

    qid = a.query_id or "query"
    write(qid, EMPTY)  # safety net first
    try:
        ensure_server()
        if not wait_ready(t0 + 12):
            return 0
        left = QUERY_BUDGET - (time.time() - t0) - 2.5
        r = call({"op": "query", "corpus": os.path.abspath(a.corpus), "query": a.query or "", "budget": left - 1}, left)
        if isinstance(r.get("answer"), str) and isinstance(r.get("citations"), list):
            write(qid, {"answer": r["answer"], "citations": [str(c) for c in r["citations"]],
                        "confidence": float(r.get("confidence", 0.0))})
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
