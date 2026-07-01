import os, json, hashlib, re, datetime
from rrr.paths import runs_path


def ensure_dir(path):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def extract_first_json(raw: str):
    """v15.13: robustly extract the first complete JSON object/array from a
    model response, tolerating preamble, trailing prose, and markdown fences.

    The old pattern — json.loads(raw[raw.find("{"):raw.rfind("}")+1]) — spans
    the first '{' to the LAST '}', which breaks whenever a model emits prose
    containing braces after the object (verbose models like qwen3:30b-a3b,
    or a frontier API model that appends an explanation). This scans for the
    first '{' or '[' and bracket-matches to its true close, respecting string
    literals and escapes, then json.loads that slice.

    Returns the parsed object, or None if no complete JSON value is found.
    """
    if not raw:
        return None
    text = raw.strip()
    # Strip a leading ```json / ``` fence if present (frontier-model habit).
    if text.startswith("```"):
        nl = text.find("\n")
        if nl != -1:
            text = text[nl + 1:]
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start = text.find(open_ch)
        if start == -1:
            continue
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(text)):
            c = text[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
                continue
            if c == '"':
                in_str = True
            elif c == open_ch:
                depth += 1
            elif c == close_ch:
                depth -= 1
                if depth == 0:
                    candidate = text[start:i + 1]
                    try:
                        return json.loads(candidate)
                    except Exception:
                        break  # first object malformed; don't try to be clever
    return None


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def save_json(obj, path):
    # v15.14: atomic write (tmp + os.replace). Run artifacts (ledger,
    # manifest, metrics, per-doc summaries) were written in place; a crash
    # mid-write left a truncated JSON that a replay/downstream reader then
    # failed to parse. os.replace is atomic on both POSIX and Windows.
    path = str(path)
    ensure_dir(os.path.dirname(path))
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def env_int(name: str, default: int) -> int:
    """v15.14: guarded int(os.environ...) parse. A malformed value used to
    raise ValueError deep inside the pipeline (or at module import time for
    writer knobs); now it warns and falls back to the default."""
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return int(str(raw).strip())
    except (TypeError, ValueError):
        print(f"[RRR] WARN: env {name}={raw!r} is not an integer; using default {default}")
        return default


def write_run(task, claim, evidence, result, subdir=None):
    base = runs_path() if not subdir else runs_path(subdir)
    ensure_dir(str(base))
    ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out = {"task": task, "topic_or_claim": claim, "evidence": evidence, "result": result, "timestamp": ts}
    fname = f"{task}_{ts}.json"
    save_json(out, str(base / fname))
    return fname


_WS = re.compile(r"\s+")
def normalize_space(s: str) -> str:
    return _WS.sub(" ", s or "").strip()
