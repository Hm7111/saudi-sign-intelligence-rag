"""Corpus walking and parsing. Every file is parsed defensively: one bad file never stops the walk.

Each parsed file becomes {"path", "text", "kind", "status", "family"}:
  * status  "current" | "withdrawn"  - a file is withdrawn only when it says so about ITSELF (file name, or a
             status line in its own header). A current revision that mentions "revision 1 is withdrawn" in its
             history is NOT withdrawn (that wording appears in the current TQ-40 datasheet).
  * family  file name without its revision token, so r1/r2 of one document can be compared.
"""
import os, csv, re, ast, io, logging

log = logging.getLogger("ingest")

TEXT_EXT = {".txt", ".log", ".md", ".rst", ".json", ".yaml", ".yml", ".ini", ".cfg", ".conf", ".toml", ".xml",
            ".html", ".htm", ".tsv"}
CODE_EXT = {".py", ".js", ".ts", ".sh", ".c", ".cc", ".cpp", ".h", ".hpp", ".java", ".go", ".rs", ".sql", ".rb"}
IMG_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".gif"}

SELF_WITHDRAWN = re.compile(r"\b(withdrawn|superseded\s+by|obsolete|deprecated|do\s+not\s+use|do\s+not\s+design|retired|"
                            r"no\s+longer\s+valid|replaced\s+by)\b", re.I)
NAME_WITHDRAWN = re.compile(r"(withdrawn|superseded|obsolete|deprecated|retired)", re.I)
REV_TOKEN = re.compile(r"(?:[_\-\s.]?(?:r|rev|v|ver|version)[_\-.]?\d+[a-z]?)", re.I)


def walk_corpus(root):
    """Return [(abs_path, rel_path)] in a stable order; never raises (unreadable dirs are skipped)."""
    out = []
    for dirpath, dirnames, filenames in os.walk(root, onerror=lambda e: log.warning("walk error %s", e)):
        dirnames.sort()
        for fn in sorted(filenames):
            p = os.path.join(dirpath, fn)
            out.append((p, os.path.relpath(p, root).replace(os.sep, "/")))
    return out


def _decode(raw):
    if raw.startswith(b"\xef\xbb\xbf"):
        return raw[3:].decode("utf-8", "replace")
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16", "replace")
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("cp1252", "replace")


def _read_bytes(path):
    with open(path, "rb") as f:  # raises PermissionError on an unreadable file - caller catches
        return f.read()


def parse_pdf(path, ocr_pdf_page=None):
    import pymupdf
    doc = pymupdf.open(path)
    try:
        if doc.needs_pass or doc.is_encrypted:
            return None  # encrypted: never a valid source, never try to read it
        parts = []
        for i, page in enumerate(doc):
            if i >= 300:
                break
            t = page.get_text("text") or ""
            if t.strip():
                parts.append(f"[page {i + 1}]\n{t.strip()}")
            elif ocr_pdf_page is not None and i < 6:  # scanned page: let the vision model read it
                pix = page.get_pixmap(dpi=150)
                txt = ocr_pdf_page(pix.tobytes("png"))
                if txt and txt.strip():
                    parts.append(f"[page {i + 1}, scanned]\n{txt.strip()}")
        return "\n".join(parts)
    finally:
        doc.close()


def _docx_blocks(d):
    """Paragraphs and tables in document order (python-docx lists them separately)."""
    from docx.table import Table
    from docx.text.paragraph import Paragraph
    body = d.element.body
    for child in body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            yield Paragraph(child, d)
        elif tag == "tbl":
            yield Table(child, d)


def _table_lines(rows):
    rows = [[re.sub(r"\s+", " ", c or "").strip() for c in r] for r in rows]
    rows = [r for r in rows if any(r)]
    if not rows:
        return []
    header = rows[0]
    lines = [" | ".join(header)]
    for r in rows[1:]:
        if len(r) == len(header) and any(header):
            lines.append(" ; ".join(f"{h}: {c}" if h else c for h, c in zip(header, r) if c))
        else:
            lines.append(" | ".join(r))
    return lines


def parse_docx(path):
    import docx
    d = docx.Document(path)
    parts = []
    try:
        for sec in d.sections:
            for p in sec.header.paragraphs:
                if p.text.strip():
                    parts.append("[header] " + p.text.strip())
    except Exception:
        pass
    ti = 0
    for b in _docx_blocks(d):
        if b.__class__.__name__ == "Paragraph":
            if b.text.strip():
                parts.append(b.text.strip())
        else:
            ti += 1
            rows = []
            for r in b.rows:
                cells, last = [], None
                for c in r.cells:  # merged cells repeat the same object
                    cells.append("" if c._tc is last else c.text)
                    last = c._tc
                rows.append(cells)
            parts.append(f"[table {ti}]\n" + "\n".join(_table_lines(rows)))
    try:
        for sec in d.sections:
            for p in sec.footer.paragraphs:
                if p.text.strip():
                    parts.append("[footer] " + p.text.strip())
    except Exception:
        pass
    return "\n".join(parts)


def parse_xlsx(path):
    import openpyxl
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    parts = []
    try:
        for ws in wb.worksheets:
            rows = [["" if c is None else (f"{c:g}" if isinstance(c, float) else str(c)) for c in r]
                    for r in ws.iter_rows(values_only=True)]
            lines = _table_lines(rows)
            if lines:
                parts.append(f"[sheet: {ws.title}]\n" + "\n".join(lines))
    finally:
        wb.close()
    return "\n".join(parts)


def parse_csv(path, delim=None):
    text = _decode(_read_bytes(path))
    if delim is None:
        try:
            delim = csv.Sniffer().sniff(text[:4096], delimiters=",;\t|").delimiter
        except Exception:
            delim = ","
    rows = list(csv.reader(io.StringIO(text), delimiter=delim))
    return "\n".join(_table_lines(rows))


def parse_code(path):
    text = _decode(_read_bytes(path))
    extra = []
    if path.lower().endswith(".py"):
        # Surface module/class constants and keyword defaults explicitly (the answer may be a default value).
        try:
            tree = ast.parse(text)
            for node in ast.walk(tree):
                if isinstance(node, (ast.Assign, ast.AnnAssign)):
                    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                    for t in targets:
                        if isinstance(t, ast.Name) and node.value is not None:
                            try:
                                extra.append(f"CONSTANT {t.id} = {ast.literal_eval(node.value)!r}")
                            except Exception:
                                pass
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    a = node.args
                    pos = a.args[-len(a.defaults):] if a.defaults else []
                    for arg, dv in list(zip(pos, a.defaults)) + list(zip(a.kwonlyargs, a.kw_defaults)):
                        if dv is None:
                            continue
                        try:
                            extra.append(f"DEFAULT {node.name}({arg.arg}) = {ast.literal_eval(dv)!r}")
                        except Exception:
                            extra.append(f"DEFAULT {node.name}({arg.arg}) = {ast.unparse(dv)}")
        except Exception:
            pass
    if extra:
        text = "[extracted constants and defaults]\n" + "\n".join(extra) + "\n[source]\n" + text
    return text


def family_of(rel_path):
    base = os.path.splitext(os.path.basename(rel_path))[0].lower()
    base = NAME_WITHDRAWN.sub("", base)
    base = REV_TOKEN.sub("", base)
    return os.path.dirname(rel_path).lower() + "/" + re.sub(r"[_\-\s.]+", "_", base).strip("_")


def revision_of(rel_path, text):
    m = re.search(r"(?:^|[_\-\s.])(?:r|rev|v|ver)[_\-.]?(\d+)", os.path.basename(rel_path), re.I)
    if m:
        return int(m.group(1))
    m = re.search(r"\b(?:revision|rev\.?|version)\s*(\d+)\b", text[:400], re.I)
    return int(m.group(1)) if m else None


def self_withdrawn(rel_path, text):
    if NAME_WITHDRAWN.search(os.path.basename(rel_path)):
        return True
    head = [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.startswith("[page")][:4]
    for ln in head:
        if re.search(r"\bsupersedes\b", ln, re.I) and not re.search(r"superseded\s+by", ln, re.I):
            continue  # "supersedes revision 1" = this IS the current one
        if SELF_WITHDRAWN.search(ln):
            return True
    return False


def parse_file(abs_path, rel_path, ocr_image=None, ocr_pdf_page=None):
    """Return {path, text, kind, status, family, revision} or None. Never raises."""
    ext = os.path.splitext(abs_path)[1].lower()
    try:
        if ext == ".pdf":
            text, kind = parse_pdf(abs_path, ocr_pdf_page), "pdf"
        elif ext in (".docx", ".docm"):
            text, kind = parse_docx(abs_path), "docx"
        elif ext in (".xlsx", ".xlsm"):
            text, kind = parse_xlsx(abs_path), "xlsx"
        elif ext == ".csv":
            text, kind = parse_csv(abs_path), "csv"
        elif ext == ".tsv":
            text, kind = parse_csv(abs_path, "\t"), "csv"
        elif ext in CODE_EXT:
            text, kind = parse_code(abs_path), "code"
        elif ext in TEXT_EXT:
            text, kind = _decode(_read_bytes(abs_path)), "text"
        elif ext in IMG_EXT:
            if ocr_image is None:
                return None
            _read_bytes(abs_path)  # permission probe before the (expensive) vision pass
            text, kind = ocr_image(abs_path), "image"
        else:
            return None  # unknown type: skip, never fail
    except PermissionError:
        log.warning("skip %s: permission denied", rel_path)
        return None
    except Exception as e:  # corrupt / encrypted office file / anything else
        log.warning("skip %s: %r", rel_path, e)
        return None
    if not text or not text.strip():
        return None
    return {"path": rel_path, "text": text.strip(), "kind": kind,
            "status": "withdrawn" if self_withdrawn(rel_path, text) else "current",
            "family": family_of(rel_path), "revision": revision_of(rel_path, text)}


def mark_superseded(docs):
    """Inside one family, any lower revision than the newest current one is superseded too."""
    fam = {}
    for d in docs:
        fam.setdefault(d["family"], []).append(d)
    for members in fam.values():
        revs = [m["revision"] for m in members if m["revision"] is not None and m["status"] == "current"]
        if len(members) > 1 and revs:
            top = max(revs)
            for m in members:
                if m["revision"] is not None and m["revision"] < top:
                    m["status"] = "withdrawn"
    return docs
