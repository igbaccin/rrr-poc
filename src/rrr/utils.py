import os, json, hashlib, re, datetime
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()
def save_json(obj, path):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
def write_run(task, claim, evidence, result, subdir=None):
    ensure_dir("runs")
    base = "runs" if not subdir else os.path.join("runs", subdir); ensure_dir(base)
    ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out = {"task": task, "topic_or_claim": claim, "evidence": evidence, "result": result, "timestamp": ts}
    fname = f"{task}_{ts}.json"
    save_json(out, os.path.join(base, fname))
    return fname
_WS = re.compile(r"\s+")
def normalize_space(s: str) -> str:
    return _WS.sub(" ", s or "").strip()
def jaccard(a_words, b_words) -> float:
    try:
        a = set(a_words); b = set(b_words)
        if not a and not b: return 1.0
        inter = len(a & b); union = len(a | b)
        return inter / union if union else 0.0
    except Exception:
        return 0.0
