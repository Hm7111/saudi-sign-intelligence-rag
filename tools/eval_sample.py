"""Score the solution on the starter-kit sample exactly like the grader (normalised answer + exact citation set).

    python3 eval_sample.py <corpus_dir> <sample-questions.json> [out.json]

Runs the real server code in-process (same model load, same index pass, same query path, minus the socket).
"""
import os, sys, json, time, re

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "app"))
import server

corpus, qfile = sys.argv[1], sys.argv[2]
out_path = sys.argv[3] if len(sys.argv) > 3 else None


def norm(s):
    return re.sub(r"[\s\-._·]", "", str(s)).upper()


t0 = time.time()
server.load_model()
if not server.STATE["ready"]:
    print("MODEL FAILED:", server.STATE["error"])
    sys.exit(1)
t_load = time.time() - t0
t1 = time.time()
info = server.do_index(os.path.abspath(corpus))
t_index = time.time() - t1
print(f"load {t_load:.1f}s, index {t_index:.1f}s: {info}")
for d in server.STATE["index"].docs:
    if d["kind"] == "image":
        print("OCR", d["path"], "->", d["text"].replace("\n", " | ")[:300])

qs = json.load(open(qfile, encoding="utf-8"))["queries"]
score, rows = 0, []
for q in qs:
    t = time.time()
    r = server.handle({"op": "query", "corpus": os.path.abspath(corpus), "query": q["query"], "budget": 26})
    dt = time.time() - t
    exp_a = q["expected_answer"]
    ok_a = norm(r.get("answer", "")) in {norm(x) for x in [exp_a] + q.get("answer_aliases", [])}
    ok_c = set(r.get("citations", [])) == set(q["expected_citations"])
    ok = ok_a and ok_c
    score += 20 * ok
    rows.append({"n": q["n"], "ok": ok, "answer": r.get("answer"), "citations": r.get("citations"), "secs": round(dt, 1),
                 "expected": exp_a, "expected_citations": q["expected_citations"]})
    print(f"Q{q['n']:>2} {'PASS' if ok else 'FAIL'} {dt:5.1f}s  ans={r.get('answer')!r} (exp {exp_a!r}) "
          f"cites={r.get('citations')} {'' if ok_c else '(exp ' + str(q['expected_citations']) + ')'}")
print(f"SCORE {score}/200   load {t_load:.0f}s  index {t_index:.0f}s  total {time.time() - t0:.0f}s")
if out_path:
    json.dump({"score": score, "load_s": t_load, "index_s": t_index, "rows": rows, "index_info": info}, open(out_path, "w"), indent=1)
