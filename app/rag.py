"""Index, retrieval and grounded answering. The model is injected (llm.generate / llm.look), so this file is testable.

Design (what is different from a plain top-k RAG):
  * Long-context when it fits: a typical graded corpus is a few dozen short files, so ALL current files go into the
    prompt (ranked), and retrieval only orders them. Nothing can be "not retrieved". Larger corpora fall back to the
    best-ranked files plus identifier hops.
  * Superseded documents are removed from the prompt when a current revision of the same family exists - they can
    never be the right citation, so the model never sees the decoy number.
  * Citations are not the model's free choice: it names the file that STATES the value (with a verbatim quote) and,
    for chains, the file that gave it the identifier it looked up. Code then checks both: the quote/value must be in
    the value file, and a link file is kept only if the shared identifier is in both files and NOT in the question.
  * Images: answered from the OCR text, then re-checked by showing the actual image to the vision model.
"""
import re, json, math, time, pickle, logging
from collections import Counter

log = logging.getLogger("rag")

FULL_CONTEXT_CHARS = 60000      # whole corpus in the prompt below this size (~15k tokens)
PER_FILE_CHARS = 9000           # cap for one file inside a ranked (non-full) context

ID_RE = re.compile(r"\b(?=[A-Za-z0-9_#-]*\d)(?=[A-Za-z0-9_#-]*[A-Za-z])[A-Za-z][A-Za-z0-9]*(?:[-_][A-Za-z0-9#]+)+\b"
                   r"|\b[A-Z]{1,4}\d{3,}[A-Z]?\b|\b[A-Z][A-Z0-9]*_[A-Z0-9_]+#?")
TOK_RE = re.compile(r"[a-z0-9#]+(?:[-_./][a-z0-9#]+)*")
STOP = set(("the a an of to in on for is are was were what which who whom when where how does do did with by at from "
            "as and or be been this that it its into than then there their this these those per any all value").split())


def norm(s):
    return re.sub(r"[\s\-._·]", "", str(s)).upper()


def stem(t):
    if len(t) > 5 and not any(ch.isdigit() for ch in t):
        for suf in ("ings", "ing", "ers", "ed", "es", "s"):
            if t.endswith(suf) and len(t) - len(suf) >= 3:
                return t[:-len(suf)]
    return t


def tokenize(text):
    toks = []
    for t in TOK_RE.findall(text.lower()):
        if t in STOP:
            continue
        t = stem(t)
        toks.append(t)
        if re.search(r"[-_./]", t):
            toks.extend(p for p in re.split(r"[-_./]", t) if p and p not in STOP)
    return toks


class BM25:
    def __init__(self, docs_tokens, k1=1.4, b=0.6):
        self.k1, self.b = k1, b
        self.tf = [Counter(t) for t in docs_tokens]
        self.len = [len(t) or 1 for t in docs_tokens]
        self.avg = sum(self.len) / max(1, len(self.len))
        df = Counter()
        for t in self.tf:
            df.update(t.keys())
        n = len(docs_tokens)
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}

    def scores(self, q_tokens):
        out = []
        for tf, ln in zip(self.tf, self.len):
            s = 0.0
            for w in set(q_tokens):
                f = tf.get(w)
                if f:
                    s += self.idf[w] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * ln / self.avg))
            out.append(s)
        return out


def chunk(text, size=1200, overlap=200):
    lines, chunks, cur = text.splitlines(), [], ""
    for ln in lines:
        if len(cur) + len(ln) + 1 > size and cur:
            chunks.append(cur)
            cur = cur[-overlap:] + "\n" + ln
        else:
            cur = (cur + "\n" + ln) if cur else ln
    if cur.strip():
        chunks.append(cur)
    return chunks


SALIENT_RE = re.compile(r"\b(warn(ing)?|error|err|fatal|crit(ical)?|fail(ed|ure)?|exception|panic|incident|alert|"
                        r"throttl\w*|timeout|denied|abort\w*|linked|ticket|caused|root cause|fixed|resolved)\b", re.I)


def salient_view(text, budget):
    """Long file (typically a log full of routine lines): keep its head, every line that carries an event (warning /
    error / incident...) or an identifier that is rare inside the file, and the tail - in original order."""
    lines = text.splitlines()
    idc = Counter()
    per_line = []
    for ln in lines:
        ids = {norm(x) for x in ID_RE.findall(ln)}
        per_line.append(ids)
        idc.update(ids)
    keep = set(range(min(6, len(lines)))) | set(range(max(0, len(lines) - 2), len(lines)))
    for i, ln in enumerate(lines):
        if SALIENT_RE.search(ln) or any(idc[x] <= 3 for x in per_line[i]):
            keep.update({i})
    out, last, size = [], -2, 0
    for i in sorted(keep):
        if i != last + 1:
            out.append("[...]")
        out.append(lines[i]); size += len(lines[i]) + 1; last = i
        if size > budget:
            out.append("[... truncated]"); break
    return "\n".join(out)


class Index:
    def __init__(self, docs):
        self.docs = docs
        for d in docs:  # compact view for long files, computed once at index time
            d["view"] = d["text"] if len(d["text"]) <= PER_FILE_CHARS else salient_view(d["text"], PER_FILE_CHARS // 2)
        current_fams = {d["family"] for d in docs if d["status"] == "current"}
        for d in docs:  # a withdrawn doc with a current sibling is never a source: hide it completely
            d["hidden"] = d["status"] == "withdrawn" and d["family"] in current_fams
        self.chunks = []
        for i, d in enumerate(docs):
            name = re.sub(r"[_\-./]", " ", d["path"])
            for c in chunk(d["text"]):
                self.chunks.append((i, name + "\n" + c))
        self.bm25 = BM25([tokenize(c) for _, c in self.chunks]) if self.chunks else None
        self.ids = []  # identifiers per doc
        for d in docs:
            self.ids.append({norm(x) for x in ID_RE.findall(d["text"]) if len(norm(x)) >= 4})

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)

    def visible(self):
        return [i for i, d in enumerate(self.docs) if not d["hidden"]]

    def rank_files(self, question):
        """File scores = best chunk + 0.3 * second best; plus identifier hops (chains)."""
        vis = set(self.visible())
        fs = {i: 0.0 for i in vis}
        if self.bm25:
            per = {}
            for (di, _), s in zip(self.chunks, self.bm25.scores(tokenize(question))):
                per.setdefault(di, []).append(s)
            for di, ss in per.items():
                if di in vis:
                    ss.sort(reverse=True)
                    fs[di] = ss[0] + 0.3 * (ss[1] if len(ss) > 1 else 0)
        qids = {norm(x) for x in ID_RE.findall(question)}
        for di in vis:  # files holding an identifier named in the question
            if qids & self.ids[di]:
                fs[di] += 5.0
        top = sorted(vis, key=lambda d: -fs[d])[:3]
        for t in top:  # second hop: files sharing a rare identifier with a top file
            for di in vis:
                if di != t and fs[t] > 0:
                    shared = {x for x in self.ids[t] & self.ids[di] if sum(x in s for s in self.ids) <= 3}
                    if shared:
                        fs[di] += 0.5 * fs[t]
        return sorted(vis, key=lambda d: -fs[d]), fs


def file_text(index, di, question):
    d = index.docs[di]
    if len(d["text"]) <= PER_FILE_CHARS:
        return d["text"]
    view = d.get("view") or d["text"][:PER_FILE_CHARS // 2]
    if index.bm25 is None:
        return view
    q = tokenize(question)
    cand = [(s, c) for (dd, c), s in zip(index.chunks, index.bm25.scores(q)) if dd == di]
    cand.sort(key=lambda x: -x[0])
    extra, size = [], 0
    for s, c in cand[:4]:
        body = c.split("\n", 1)[1] if "\n" in c else c
        if s > 0 and body not in view and size + len(body) <= PER_FILE_CHARS // 2:
            extra.append(body); size += len(body)
    return view + ("\n[best matching passages]\n" + "\n...\n".join(extra) if extra else "")


def build_context(index, order, question=""):
    texts = {di: file_text(index, di, question) for di in order}
    total = sum(len(t) for t in texts.values())
    labels, blocks = {}, []
    budget = FULL_CONTEXT_CHARS
    for di in order:
        d = index.docs[di]
        text = texts[di] if total <= FULL_CONTEXT_CHARS else texts[di][:PER_FILE_CHARS]
        if blocks and budget - len(text) < 0:
            break
        budget -= len(text)
        lab = f"F{len(labels) + 1}"
        labels[lab] = di
        kind = {"image": "image (text read from the picture)", "code": "source code"}.get(d["kind"], d["kind"])
        status = "WITHDRAWN - superseded, never cite" if d["status"] == "withdrawn" else "current"
        blocks.append(f'<file id="{lab}" path="{d["path"]}" type="{kind}" status="{status}">\n{text}\n</file>')
    return "\n\n".join(blocks), labels


PROMPT = """You answer questions about a FICTIONAL company using ONLY the files below. Anything you remember from outside these files is wrong by definition.

{context}

QUESTION: {question}

How to answer:
1. Find the file that explicitly STATES the answer value. Read tables row by row (a row is "column: value ; column: value"). In source code the current constant / default value is the answer, not an old value mentioned in a comment. A file marked WITHDRAWN is never the source.
2. Chains: if the question does not name the item directly (e.g. "the incident in the log", "the warning on node 12") and you had to read an identifier (ticket, error code, article id, part number, serial) in one file and then look that identifier up in another file - possibly again in a third file - every file of that chain is needed: the value file plus each link file that handed you an identifier. If the question already contains the identifier, the value file alone is enough.
3. Cite nothing else: a file that only discusses the same topic, or repeats the value, is NOT needed.
4. "answer" is the value only, complete with its qualifiers (e.g. "Q3 FY27", "REV-C2", "ORR-FAN-2214-B", "4.3.2"). For a plain quantity give just the number without the unit (e.g. "94", "180"). No sentence.
5. If no file states the answer, the answer is "" - never guess, never compute a value that is not written.

Reply with ONE JSON object only:
{{"reasoning": "one or two short sentences", "answer": "...", "value_file": "F?", "value_quote": "exact words copied from the value file that contain the answer", "link_files": ["F? for each link file of a chain, in order; empty list if none"], "link_ids": ["the identifier each link file handed on"]}}"""


def parse_json(text):
    text = re.sub(r"```(?:json)?", "", text)
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    s = m.group(0)
    for cand in (s, s.replace("'", '"')):
        try:
            return json.loads(cand)
        except Exception:
            pass
    out = {}  # last resort: field-by-field
    for k in ("answer", "value_file", "value_quote"):
        mm = re.search(rf'"{k}"\s*:\s*"((?:[^"\\]|\\.)*)"', s)
        if mm:
            out[k] = mm.group(1)
    return out or None


REFUSALS = {"", "NONE", "NA", "N/A", "UNKNOWN", "NOTFOUND", "NULL", "NOTSTATED", "NOTAVAILABLE", "NOTINTHEFILES"}
EMPTY = {"answer": "", "citations": [], "confidence": 0.0}


def _quote_in(quote, text):
    q, t = norm(quote), norm(text)
    if q and q in t:
        return True
    words = [w for w in re.findall(r"[A-Za-z0-9#]+", quote) if len(w) > 1]
    return bool(words) and sum(norm(w) in t for w in words) / len(words) >= 0.8


def answer(index, question, llm, deadline):
    order, fs = index.rank_files(question)
    if not order:
        return dict(EMPTY)
    context, labels = build_context(index, order, question)
    raw = llm.generate(PROMPT.format(context=context, question=question), max_new_tokens=260)
    obj = parse_json(raw) or {}
    log.info("llm: %s", raw.replace("\n", " ")[:600])
    ans = re.sub(r"\s+", " ", str(obj.get("answer", "") or "")).strip().strip('"').strip()
    if norm(ans) in REFUSALS or len(ans) > 60 or len(ans.split()) > 6:
        return dict(EMPTY)
    docs = index.docs
    vf = labels.get(str(obj.get("value_file", "")).strip())
    quote = str(obj.get("value_quote", "") or "")
    na = norm(ans)

    def has_value(di):
        return di is not None and na and na in norm(docs[di]["text"])

    # 1) the value file must really contain the value (or the quoted words that carry it)
    if not (has_value(vf) or (vf is not None and quote and _quote_in(quote, docs[vf]["text"]) and
                              any(norm(w) in norm(docs[vf]["text"]) for w in ans.split()))):
        holders = [di for di in labels.values() if has_value(di)]
        if not holders:
            img = _image_check(index, question, labels, llm, ans, deadline)
            if img is None:
                return dict(EMPTY)
            vf, ans = img
        else:
            holders.sort(key=lambda d: -fs.get(d, 0))
            vf = holders[0]
    # 2) an image source is re-read from the picture itself (OCR text can drift)
    if docs[vf]["kind"] == "image" and time.time() < deadline - 6:
        seen = llm.look(docs[vf]["abs_path"], _look_prompt(question), max_new_tokens=40)
        seen = _clean_value(seen)
        if seen and norm(seen) != norm(ans) and norm(seen) in norm(docs[vf]["text"]):
            ans = seen
    cites = [vf]
    # 3) chain: a link file is kept only if it really hands on an identifier (shared with the value file or with an
    #    already accepted link) that the question itself does not contain
    raw_links = obj.get("link_files", obj.get("link_file", [])) or []
    raw_links = [raw_links] if isinstance(raw_links, str) else raw_links
    links = [labels[str(x).strip()] for x in raw_links if str(x).strip() in labels and labels[str(x).strip()] != vf]
    lids = obj.get("link_ids", obj.get("link_id", [])) or []
    lids = {norm(x) for x in ([lids] if isinstance(lids, str) else lids) if norm(x)}
    q_norm, q_ids = norm(question), {norm(x) for x in ID_RE.findall(question)}

    def hands_on(a, b):
        shared = index.ids[a] & index.ids[b]
        shared |= {x for x in lids if x in norm(docs[a]["text"]) and x in norm(docs[b]["text"])}
        return any(x not in q_ids and x not in q_norm and x != na for x in shared)

    accepted, changed = [], True
    while changed:
        changed = False
        for lf in links:
            if lf not in accepted and any(hands_on(lf, t) for t in [vf] + accepted):
                accepted.append(lf); changed = True
    cites += accepted
    return {"answer": ans, "citations": [docs[d]["path"] for d in cites], "confidence": 0.9}


def _look_prompt(question):
    return ("Look at this image and answer the question using ONLY text printed in the image. "
            "Reply with the value only (no sentence). If the image does not show it, reply NONE.\n"
            f"Question: {question}")


def _clean_value(s):
    s = (s or "").strip().strip('"').strip().splitlines()[0] if (s or "").strip() else ""
    s = re.sub(r"^(answer|value)\s*[:\-]\s*", "", s, flags=re.I).strip().rstrip(".")
    return "" if norm(s) in REFUSALS or len(s) > 60 else s


def _image_check(index, question, labels, llm, ans, deadline):
    """The model gave a value no text holds: accept it only if an image in the context visibly shows it."""
    for di in labels.values():
        d = index.docs[di]
        if d["kind"] != "image" or time.time() > deadline - 6:
            continue
        seen = _clean_value(llm.look(d["abs_path"], _look_prompt(question), max_new_tokens=40))
        if seen and (norm(seen) == norm(ans) or norm(ans) in norm(seen)):
            return di, seen
    return None
