from typing import List
import os, subprocess, json, re, ast, hashlib, time
from rrr.utils import ensure_dir
from rrr.stance import classify_evidence_stance
from rrr.metrics import RunMetrics
from rrr.manifest import write_run_manifest
from rrr.paths import runs_path, logs_path
from rrr.text import tokenize
from rapidfuzz import fuzz

_MODEL      = os.environ.get("RRR_MODEL", "mistral")
_OPTIONS_REASON = {
    "temperature": 0.0,
    "num_ctx": int(os.environ.get("RRR_REASONER_CTX", "8192")),
    "num_predict": int(os.environ.get("RRR_REASONER_PRED", "2000")),
}
_OPTIONS_MECHANISM = {
    "temperature": 0.0,
    "num_ctx": int(os.environ.get("RRR_MECH_CTX", "4096")),
    "num_predict": int(os.environ.get("RRR_MECH_PRED", "300")),
}
_OPTIONS_CLUSTER = {
    "temperature": 0.0,
    "num_ctx": int(os.environ.get("RRR_CLUSTER_CTX", "8192")),
    "num_predict": int(os.environ.get("RRR_CLUSTER_PRED", "2500")),
}
_KEEP_ALIVE = "30m"
_MECHANISM_PROMPT_VERSION = os.environ.get("RRR_MECH_PROMPT_VERSION", "2026-06-25-batch1")


class ClusteringFailedError(Exception):
    """Raised when clustering fails after all retries - triggers full pipeline restart."""
    pass


def _build_prompt(evidence_texts: List[str], claim: str) -> str:
    prompt = (
        "You are an economic historian.\n"
        "Given the following evidence snippets, answer the claim ONLY using the retrieved text. Do not add external facts.\n\n"
        f"Claim:\n{claim}\n\n"
        "Evidence (page-bounded extracts):\n"
        + "\n\n---\n\n".join(evidence_texts)
        + "\n\n"
    )
    prompt += (
        "Task: Produce ONE and ONLY ONE valid JSON object following this schema EXACTLY:\n\n"
        "{\n"
        '  "topic": "<string>",\n'
        '  "positions": [\n'
        '    {\n'
        '      "label": "<short label>",\n'
        '      "mechanism_summary": "<1-3 sentence summary>",\n'
        '      "supporting_docs": ["DOCID", ...],\n'
        '      "representative_quotes": [ { "doc_id":"DOCID", "page": X, "quote":"exact substring" }, ... ],\n'
        '      "points_of_dispute": ["short bullet strings"]\n'
        '    }\n'
        '  ],\n'
        '  "unrepresented_docs": ["DOCID", ...],\n'
        '  "notes": "<short note or empty string>"\n'
        "}\n\n"
        "Strict output rules:\n"
        "- The entire reply must be a SINGLE JSON object. No commentary, no explanations, no preambles.\n"
        "- If you cannot fill a field, use an empty string or empty array [] - never omit the key.\n"
        "- Do not add text before or after the JSON. The system will reject any non-JSON tokens.\n\n"
        "After printing the JSON, output the word SUMMARY on a new line and then write your 2â€“6-sentence summary.\n"
    )
    return prompt

def _extract_json_and_summary(raw: str):
    start = raw.find('{')
    if start == -1:
        return None, raw.strip()
    candidate = raw[start:]
    try:
        obj = json.loads(candidate)
        pretty = json.dumps(obj, indent=2, ensure_ascii=False)
        remainder = ""
        if "SUMMARY" in candidate:
            remainder = candidate.split("SUMMARY", 1)[-1].strip()
        return pretty, remainder
    except json.JSONDecodeError:
        pass
    fixed = re.sub(r"[^{}]*$", "", candidate)
    opens, closes = fixed.count("{"), fixed.count("}")
    if opens > closes:
        fixed += "}" * (opens - closes)
    try:
        obj = json.loads(fixed)
        pretty = json.dumps(obj, indent=2, ensure_ascii=False)
        remainder = ""
        if "SUMMARY" in candidate:
            remainder = candidate.split("SUMMARY", 1)[-1].strip()
        return pretty, remainder
    except Exception:
        try:
            obj = ast.literal_eval(fixed)
            pretty = json.dumps(obj, indent=2, ensure_ascii=False)
            remainder = ""
            if "SUMMARY" in candidate:
                remainder = candidate.split("SUMMARY", 1)[-1].strip()
            return pretty, remainder
        except Exception:
            ensure_dir(str(logs_path()))
            with open(logs_path("invalid_json.txt"), "w", encoding="utf-8") as f:
                f.write(raw)
            return None, raw.strip()


def parse_reasoned_json(raw: str):
    pretty, _summary = _extract_json_and_summary(raw or "")
    if not pretty:
        return None
    try:
        return json.loads(pretty)
    except Exception:
        return None

def reason_over_evidence(evidence_texts: List[str], claim: str, model: str = _MODEL, metrics=None) -> str:
    if not evidence_texts:
        return "No evidence to reason over."
    prompt = _build_prompt(evidence_texts, claim)
    start = time.perf_counter()
    try:
        import ollama
        res = ollama.chat(model=model,
                          messages=[{"role":"user","content":prompt}],
                          options=_OPTIONS_REASON, keep_alive=_KEEP_ALIVE, stream=False)
        out = res["message"]["content"].strip()
        if metrics:
            metrics.record_llm("reasoner", model, options=_OPTIONS_REASON,
                               duration_s=time.perf_counter() - start,
                               prompt_chars=len(prompt), response_chars=len(out))
    except Exception as e_client:
        try:
            sub_start = time.perf_counter()
            p = subprocess.run(["ollama","run", model],
                               input=prompt.encode("utf-8"),
                               capture_output=True, timeout=600)
            out = p.stdout.decode("utf-8").strip() or "(no output)"
            if metrics:
                metrics.record_llm("reasoner_subprocess", model,
                                   duration_s=time.perf_counter() - sub_start,
                                   prompt_chars=len(prompt), response_chars=len(out))
        except Exception as e_sub:
            if metrics:
                metrics.record_llm("reasoner", model, options=_OPTIONS_REASON,
                                   success=False, duration_s=time.perf_counter() - start,
                                   prompt_chars=len(prompt),
                                   error=f"client {e_client}; fallback {e_sub}")
            return f"[reasoner error: client {e_client} ; fallback {e_sub}]"
    pretty_json, summary = _extract_json_and_summary(out)
    if pretty_json:
        if summary:
            summary = summary.strip()
            if not summary.lower().startswith("summary"):
                summary = "SUMMARY\n\n" + summary
            return pretty_json + "\n\n" + summary
        else:
            return pretty_json
    else:
        return out

def _mechanism_signature(topic: str, quotes, model: str = _MODEL, prompt_version: str = _MECHANISM_PROMPT_VERSION) -> str:
    h = hashlib.sha256()
    h.update(f"model={model}|prompt_version={prompt_version}\n".encode("utf-8"))
    h.update((topic or "").encode("utf-8"))
    for q in quotes:
        h.update(f"{q.get('doc_id','')}|{q.get('page','')}|{q.get('text','')}\n".encode("utf-8"))
    return h.hexdigest()[:16]

def _mechanism_cache_path(doc_id: str, sig: str) -> str:
    return str(runs_path("cache", "mechanisms", f"{doc_id}_{sig}.json"))

def _load_mechanism_cache(doc_id: str, sig: str):
    try:
        with open(_mechanism_cache_path(doc_id, sig), encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def _save_mechanism_cache(doc_id: str, sig: str, obj):
    ensure_dir(str(runs_path("cache", "mechanisms")))
    with open(_mechanism_cache_path(doc_id, sig), "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def _try_parse_cluster_json(raw: str, n_mechs: int, all_mechs: list):
    """Attempt to parse clustering JSON. Returns mech_to_cluster dict or None."""
    start = raw.find('{')
    end = raw.rfind('}') + 1
    if start == -1 or end == 0:
        return None
    
    json_str = raw[start:end]
    json_str = re.sub(r',\s*}', '}', json_str)
    json_str = re.sub(r',\s*]', ']', json_str)
    json_str = re.sub(r'(\d)\s*\n\s*"', r'\1],\n"', json_str)
    
    try:
        clusters = json.loads(json_str)
    except json.JSONDecodeError:
        return None
    
    if not isinstance(clusters, dict):
        return None
    
    mech_to_cluster = {}
    for label, indices in clusters.items():
        if not isinstance(indices, list):
            continue
        for idx in indices:
            if isinstance(idx, int) and 1 <= idx <= n_mechs:
                mech_to_cluster[all_mechs[idx - 1]] = label
    
    # Check we got reasonable coverage
    if len(mech_to_cluster) < n_mechs * 0.5:
        return None
    
    return mech_to_cluster


def _label_from_tokens(tokens):
    if not tokens:
        return "Other"
    words = []
    for tok in tokens:
        if tok not in words:
            words.append(tok)
        if len(words) >= 4:
            break
    return " ".join(w.capitalize() for w in words) or "Other"


def _fallback_cluster_mechanisms(all_mechs: list, topic: str, metrics=None) -> dict:
    topic_tokens = set(tokenize(topic))
    clusters = []
    mech_to_cluster = {}

    for mech in all_mechs:
        toks = set(tokenize(mech)) - topic_tokens
        if not toks:
            toks = set(tokenize(mech))
        best_idx = None
        best_score = 0.0
        for idx, cluster in enumerate(clusters):
            denom = len(toks | cluster["tokens"]) or 1
            score = len(toks & cluster["tokens"]) / denom
            if score > best_score:
                best_idx = idx
                best_score = score
        if best_idx is None or (best_score < 0.18 and len(clusters) < 8):
            clusters.append({"tokens": set(toks), "items": [mech]})
        else:
            clusters[best_idx]["tokens"].update(toks)
            clusters[best_idx]["items"].append(mech)

    for cluster in clusters:
        counts = {}
        for item in cluster["items"]:
            for tok in tokenize(item):
                if tok not in topic_tokens:
                    counts[tok] = counts.get(tok, 0) + 1
        ranked = sorted(counts, key=lambda t: (-counts[t], t))
        label = _label_from_tokens(ranked)
        for item in cluster["items"]:
            mech_to_cluster[item] = label

    if metrics:
        metrics.inc("cluster_fallbacks")
        metrics.set("cluster_mode", "local_token_overlap")
        metrics.set("cluster_fallback_cluster_count", len(clusters))
    return mech_to_cluster


def _cluster_mechanisms(doc_summaries: list, topic: str, metrics=None) -> dict:
    """
    Cluster mechanisms into themes.

    Uses the LLM clusterer first, then falls back to deterministic token-overlap
    clusters so a clustering failure does not restart the whole pipeline.
    """
    mech_to_docs = {}
    for doc in doc_summaries:
        did = doc.get("doc_id", "")
        for m in doc.get("mechanisms", []):
            m = (m or "").strip()
            if m:
                if m not in mech_to_docs:
                    mech_to_docs[m] = []
                mech_to_docs[m].append(did)
    
    if not mech_to_docs:
        if metrics:
            metrics.set("cluster_mode", "none")
        return {}
    
    all_mechs = list(mech_to_docs.keys())
    n_mechs = len(all_mechs)
    
    if n_mechs <= 3:
        if metrics:
            metrics.set("cluster_mode", "identity")
        return {m: m[:60] for m in all_mechs}
    
    cluster_prompt = (
        "You are a historian organizing a literature review.\n\n"
        f"Topic: {topic}\n\n"
        f"Below are {n_mechs} mechanism claims. Group them into thematic clusters.\n\n"
        "RULES:\n"
        "- Use between 4 and 8 clusters.\n"
        "- Cluster labels: SHORT (3-6 words).\n"
        f"- Valid indices are 1 to {n_mechs} only.\n"
        "- Every index must appear exactly once.\n"
        "- Output ONLY a JSON object, no other text.\n\n"
        "MECHANISMS:\n"
    )
    for i, m in enumerate(all_mechs, 1):
        m_short = m[:100] + "..." if len(m) > 100 else m
        cluster_prompt += f"{i}. {m_short}\n"
    
    cluster_prompt += (
        f"\nReturn ONLY valid JSON. Indices must be 1-{n_mechs}.\n"
        'Example: {"Theme A": [1, 2, 5], "Theme B": [3, 4]}\n'
        "JSON:\n"
    )
    
    MAX_RETRIES = 5
    mech_to_cluster = None
    
    for attempt in range(MAX_RETRIES):
        try:
            import ollama
            start = time.perf_counter()
            res = ollama.chat(
                model=_MODEL,
                messages=[{"role": "user", "content": cluster_prompt}],
                options=_OPTIONS_CLUSTER,
                keep_alive=_KEEP_ALIVE,
                stream=False
            )
            raw = res["message"]["content"].strip()
            if metrics:
                metrics.record_llm("cluster", _MODEL, options=_OPTIONS_CLUSTER,
                                   duration_s=time.perf_counter() - start,
                                   prompt_chars=len(cluster_prompt),
                                   response_chars=len(raw))
            
            ensure_dir(str(logs_path()))
            with open(logs_path(f"cluster_raw_attempt{attempt+1}.txt"), "w", encoding="utf-8") as f:
                f.write(raw)
            
            mech_to_cluster = _try_parse_cluster_json(raw, n_mechs, all_mechs)
            
            if mech_to_cluster is not None:
                break
            
            print(f"[Clustering] Attempt {attempt+1}/{MAX_RETRIES} failed to parse JSON, retrying...")
            
        except Exception as e:
            print(f"[Clustering] Attempt {attempt+1}/{MAX_RETRIES} error: {e}")
            if metrics:
                metrics.record_llm("cluster", _MODEL, options=_OPTIONS_CLUSTER,
                                   success=False, error=e)
            continue
    
    if mech_to_cluster is None:
        print(f"[Clustering] LLM clustering failed after {MAX_RETRIES} attempts; using deterministic fallback")
        mech_to_cluster = _fallback_cluster_mechanisms(all_mechs, topic, metrics=metrics)
    elif metrics:
        metrics.set("cluster_mode", "llm")
    
    for m in all_mechs:
        if m not in mech_to_cluster:
            mech_to_cluster[m] = "Other"
    
    n_clusters = len(set(mech_to_cluster.values()))
    n_assigned = sum(1 for m in all_mechs if mech_to_cluster.get(m) != "Other")
    print(f"[Clustering] {n_mechs} mechanisms -> {n_clusters} clusters ({n_assigned} assigned, {n_mechs - n_assigned} to Other)")
    return mech_to_cluster

def _build_author_year_lookup(allowed_docs):
    """Build reverse lookup: (author, year) -> doc_id for academic citation matching."""
    author_year_to_docid = {}
    for did in allowed_docs:
        clean = did.replace("EtAl", "").replace("&", "")
        parts = clean.split("_")
        if len(parts) >= 2:
            author = parts[0].lower()
            year = parts[-1].rstrip('abcdefgh')
            author_year_to_docid[(author, year)] = did
            if "EtAl" in did:
                author_year_to_docid[(author + " et al", year)] = did
    return author_year_to_docid

def _collect_cited_docs(text: str, allowed_docs, author_year_to_docid):
    """Collect cited doc_ids from both correct and academic citation formats."""
    cited_docs = set()
    
    for m in re.finditer(r"\(([A-Za-z0-9_&.\-]+):\s*p\.(\d+)\)", text):
        did = m.group(1)
        if did in allowed_docs:
            cited_docs.add(did)
    
    for m in re.finditer(r"\(([A-Za-z0-9_&]+_\d{4}[a-z]?)\)", text):
        did = m.group(1)
        if did in allowed_docs:
            cited_docs.add(did)
    
    for m in re.finditer(r"\(([A-Za-z&]+(?:\s+et\s+al\.?)?)[,\s]+(\d{4})\)", text):
        author = m.group(1).lower().strip().rstrip('.')
        year = m.group(2)
        did = author_year_to_docid.get((author, year))
        if did:
            cited_docs.add(did)
    
    for m in re.finditer(r"([A-Za-z&]+(?:\s+et\s+al\.?)?)\s+\((\d{4})\)", text):
        author = m.group(1).lower().strip().rstrip('.')
        year = m.group(2)
        did = author_year_to_docid.get((author, year))
        if did:
            cited_docs.add(did)
    
    return cited_docs

def _clean_latex(s: str) -> str:
    """Clean LaTeX artifacts from BibTeX strings."""
    if not s:
        return s
    s = s.replace('{', '').replace('}', '')
    replacements = [
        (r"\\'e", 'Ã©'), (r"\\`e", 'Ã¨'), (r'\\"e', 'Ã«'), (r'\\^e', 'Ãª'),
        (r"\\'a", 'Ã¡'), (r"\\`a", 'Ã '), (r'\\"a', 'Ã¤'), (r'\\^a', 'Ã¢'),
        (r"\\'o", 'Ã³'), (r"\\`o", 'Ã²'), (r'\\"o', 'Ã¶'), (r'\\^o', 'Ã´'),
        (r"\\'u", 'Ãº'), (r"\\`u", 'Ã¹'), (r'\\"u', 'Ã¼'), (r'\\^u', 'Ã»'),
        (r"\\'i", 'Ã­'), (r"\\`i", 'Ã¬'), (r'\\"i', 'Ã¯'), (r'\\^i', 'Ã®'),
        (r'\\c{c}', 'Ã§'), (r'\\c{C}', 'Ã‡'),
        (r'\\c{s}', 'ÅŸ'), (r'\\c{S}', 'Åž'),
        (r'\\v{s}', 'Å¡'), (r'\\v{S}', 'Å '),
        (r'\\~n', 'Ã±'), (r'\\~N', 'Ã‘'),
        (r'\\ss', 'ÃŸ'),
        (r"\\'", ''), (r'\\`', ''), (r'\\"', ''), (r'\\^', ''),
        (r'\\c', ''), (r'\\v', ''), (r'\\~', ''),
    ]
    for latex, char in replacements:
        s = s.replace(latex, char)
    s = re.sub(r'\\([a-zA-Z])', r'\1', s)
    return s.strip()

def _cite_harvard(row):
    """Format citation in Harvard (LUSEM) style."""
    def s(x):
        val = str(x).strip() if x is not None else ""
        return "" if val.lower() == "nan" else val
    
    def clean_num(x):
        val = s(x)
        if val.endswith('.0'):
            return val[:-2]
        return val
    
    author_full = _clean_latex(s(row.get("author_full")))
    authors_short = _clean_latex(s(row.get("authors")))
    title = _clean_latex(s(row.get("title")))
    year = s(row.get("year"))
    venue = _clean_latex(s(row.get("venue")))
    volume = clean_num(row.get("volume"))
    number = clean_num(row.get("number"))
    pages = s(row.get("pages"))
    
    if author_full:
        author_parts = [p.strip() for p in author_full.split(";") if p.strip()]
        formatted_authors = []
        for ap in author_parts:
            if "," in ap:
                surname, first = ap.split(",", 1)
                initials = "".join([n[0] + "." for n in first.strip().split() if n])
                formatted_authors.append(f"{surname.strip()}, {initials}")
            else:
                formatted_authors.append(ap)
        
        if len(formatted_authors) == 1:
            author_str = formatted_authors[0]
        elif len(formatted_authors) == 2:
            author_str = f"{formatted_authors[0]} and {formatted_authors[1]}"
        else:
            author_str = ", ".join(formatted_authors[:-1]) + f" and {formatted_authors[-1]}"
    else:
        author_str = authors_short or "[Unknown]"
    
    cite = f"{author_str} ({year})" if year else author_str
    
    if title:
        cite += f" '{title}'"
    
    if venue:
        cite += f", {venue}"
        if volume:
            cite += f", {volume}"
            if number:
                cite += f"({number})"
        if pages:
            cite += f", pp. {pages.replace('--', '-')}"
    
    cite += "."
    return cite


def _plan_probes(plan_obj: dict, topic: str, score_query: str) -> list:
    probes = []
    for item in plan_obj.get("probes", []) or []:
        text = re.sub(r"\s+", " ", str(item or "").strip())
        if text and text not in probes:
            probes.append(text)
    for item in [score_query, topic]:
        text = re.sub(r"\s+", " ", str(item or "").strip())
        if text and text not in probes:
            probes.append(text)
    return probes[:8] or [topic]


def _doc_admit_signature(topic: str, probes: list, doc_ids: list, settings: dict) -> str:
    h = hashlib.sha256()
    h.update((topic or "").encode("utf-8"))
    h.update(json.dumps(probes, sort_keys=True).encode("utf-8"))
    h.update(json.dumps(doc_ids, sort_keys=True).encode("utf-8"))
    h.update(json.dumps(settings, sort_keys=True).encode("utf-8"))
    return h.hexdigest()[:16]


def _doc_admit_cache_path(sig: str):
    return runs_path("cache", "doc_admit", f"{sig}.json")


def _load_doc_admit_cache(sig: str):
    obj = _load_doc_admit_cache_obj(sig)
    if not obj:
        return None
    docs = obj.get("docs", [])
    return docs if isinstance(docs, list) else None


def _load_doc_admit_cache_obj(sig: str):
    try:
        with open(_doc_admit_cache_path(sig), encoding="utf-8") as f:
            obj = json.load(f)
        docs = obj.get("docs", [])
        return obj if isinstance(docs, list) else None
    except Exception:
        return None


def _save_doc_admit_cache(sig: str, docs: list, meta: dict, rejections: list = None):
    ensure_dir(str(runs_path("cache", "doc_admit")))
    with open(_doc_admit_cache_path(sig), "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "docs": docs, "rejections": rejections or []}, f, indent=2, ensure_ascii=False)


def _best_probe_for_sentence(sentence: str, probes: list) -> str:
    if not probes:
        return ""
    return max(probes, key=lambda p: fuzz.token_set_ratio(sentence, p))


def _rerank_quotes_for_diversity(quotes: list, cap: int) -> list:
    if cap <= 0 or len(quotes) <= cap:
        return sorted(quotes, key=lambda x: x.get("score", 0), reverse=True)
    remaining = sorted(quotes, key=lambda x: x.get("score", 0), reverse=True)
    chosen = []
    seen_pages = set()
    seen_texts = []
    while remaining and len(chosen) < cap:
        best_idx = 0
        best_value = None
        for idx, q in enumerate(remaining):
            text = q.get("text", "")
            page_penalty = 8 if q.get("page") in seen_pages else 0
            similarity_penalty = max((fuzz.token_set_ratio(text, s) for s in seen_texts), default=0) * 0.10
            probe_bonus = 3 if q.get("best_probe") and q.get("best_probe") not in {x.get("best_probe") for x in chosen} else 0
            value = float(q.get("score", 0)) + probe_bonus - page_penalty - similarity_penalty
            if best_value is None or value > best_value:
                best_idx = idx
                best_value = value
        q = remaining.pop(best_idx)
        chosen.append(q)
        seen_pages.add(q.get("page"))
        seen_texts.append(q.get("text", ""))
    return chosen


def _select_budget_docs(docs: list, budget: int, probes: list, metrics=None) -> list:
    if budget <= 0 or len(docs) <= budget:
        if metrics:
            metrics.set("doc_budget_requested", budget)
            metrics.set("doc_budget_selected", len(docs))
            metrics.set("doc_budget_exhaustive", True)
        return sorted(docs, key=lambda x: x.get("avg_score", 0), reverse=True)

    selected = []
    remaining = sorted(docs, key=lambda x: x.get("avg_score", 0), reverse=True)
    covered_probes = set()

    while remaining and len(selected) < budget:
        best_idx = 0
        best_value = None
        selected_doc_ids = {d.get("doc_id") for d in selected}
        for idx, doc in enumerate(remaining):
            doc_probes = set(doc.get("probe_hits", []))
            new_probe_bonus = 5 * len(doc_probes - covered_probes)
            score = float(doc.get("avg_score", 0))
            evidence_bonus = min(len(doc.get("quotes", [])), 8)
            duplicate_penalty = 20 if doc.get("doc_id") in selected_doc_ids else 0
            value = score + new_probe_bonus + evidence_bonus - duplicate_penalty
            if best_value is None or value > best_value:
                best_idx = idx
                best_value = value
        doc = remaining.pop(best_idx)
        selected.append(doc)
        covered_probes.update(doc.get("probe_hits", []))

    selected = sorted(selected, key=lambda x: x.get("avg_score", 0), reverse=True)
    if metrics:
        metrics.set("doc_budget_requested", budget)
        metrics.set("doc_budget_selected", len(selected))
        metrics.set("doc_budget_exhaustive", False)
        metrics.set("doc_budget_probe_coverage", len(covered_probes))
        metrics.inc("docs_budget_dropped", max(0, len(docs) - len(selected)))
    return selected


def _assign_evidence_ids(doc_summaries: list):
    counter = 1
    for doc in sorted(doc_summaries, key=lambda d: d.get("doc_id", "")):
        for q in doc.get("quotes", []) or []:
            q["evidence_id"] = f"E{counter:04d}"
            counter += 1
    return counter - 1


def _mean_score(docs: list) -> float:
    vals = [float(d.get("avg_score", 0) or 0) for d in docs or []]
    return round(sum(vals) / len(vals), 2) if vals else 0.0


def _compute_topic_fit(topic: str, probes: list, all_doc_ids: list, admitted_docs: list,
                       selected_docs: list, doc_summaries: list = None, rejections: list = None):
    total = len(all_doc_ids) or 1
    represented = doc_summaries or []
    admitted_probe_hits = set()
    selected_probe_hits = set()
    for doc in admitted_docs or []:
        admitted_probe_hits.update(doc.get("probe_hits", []) or [])
    for doc in selected_docs or []:
        selected_probe_hits.update(doc.get("probe_hits", []) or [])

    probe_count = len(probes) or 1
    rejection_counts = {}
    for item in rejections or []:
        reason = item.get("reason", "unknown")
        rejection_counts[reason] = rejection_counts.get(reason, 0) + 1

    summary = {
        "topic": topic,
        "docs_total": len(all_doc_ids),
        "docs_admitted": len(admitted_docs or []),
        "docs_selected_for_llm": len(selected_docs or []),
        "docs_represented": len(represented),
        "admission_share": round(len(admitted_docs or []) / total, 4),
        "selection_share": round(len(selected_docs or []) / total, 4),
        "represented_share": round(len(represented) / total, 4),
        "admitted_probe_coverage": round(len(admitted_probe_hits) / probe_count, 4),
        "selected_probe_coverage": round(len(selected_probe_hits) / probe_count, 4),
        "mean_admitted_score": _mean_score(admitted_docs),
        "mean_selected_score": _mean_score(selected_docs),
        "rejection_counts": rejection_counts,
        "warnings": [],
    }

    if summary["admission_share"] < 0.25:
        summary["warnings"].append("low_admitted_document_share")
    if summary["selected_probe_coverage"] < 0.5:
        summary["warnings"].append("narrow_probe_coverage")
    if summary["mean_selected_score"] and summary["mean_selected_score"] < 45:
        summary["warnings"].append("low_mean_evidence_score")
    if represented and summary["represented_share"] < 0.15:
        summary["warnings"].append("low_represented_document_share")
    return summary


def _write_json_run(name: str, obj):
    ensure_dir(str(runs_path()))
    with open(runs_path(name), "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _retrieve_doc_with_probes(retrieve_fn, doc_id: str, probes: list, topk: int, metrics=None) -> list:
    merged = {}
    per_probe_topk = max(1, topk)
    for probe in probes:
        candidates = retrieve_fn(probe, topk=per_probe_topk, doc_id=doc_id)
        if metrics:
            metrics.inc("retrieval_probe_calls")
            metrics.inc("retrieved_pages_raw", len(candidates))
        for c in candidates:
            key = (c.get("doc_id"), c.get("page"))
            score = float(c.get("bm25_score", 0.0) or 0.0)
            existing = merged.get(key)
            if not existing or score > float(existing.get("bm25_score", 0.0) or 0.0):
                item = dict(c)
                item["matched_probes"] = [probe]
                merged[key] = item
            else:
                existing.setdefault("matched_probes", []).append(probe)
    ranked = sorted(merged.values(), key=lambda x: float(x.get("bm25_score", 0.0) or 0.0), reverse=True)
    return ranked[:max(1, topk)]


def _layered_t2_inner(args, meta_path, restart_attempt=0):
    """
    Inner implementation of layered_t2.
    """
    import os, json, threading
    import pandas as pd
    from rrr.retrieve import retrieve
    from rrr.evidence_filter import select_sentences
    from rrr.validate import validate_evidence_verbose
    from rrr.utils import ensure_dir, write_run, normalize_space

    topic = args.topic
    metrics = RunMetrics("T2_LAYERED_GLOBAL", topic)
    metrics.set("restart_attempt", restart_attempt)

    with metrics.stage("load_metadata"):
        df = pd.read_csv(meta_path)
        df["doc_id"] = df["doc_id"].astype(str)

    refs = {str(r["doc_id"]): _cite_harvard(r) for _, r in df.iterrows()}
    all_doc_ids = df["doc_id"].tolist()
    metrics.set("metadata_path", str(meta_path))
    metrics.set("docs_total", len(all_doc_ids))

    ensure_dir(str(runs_path()))
    ensure_dir(str(runs_path("layered_docs")))

    PER_DOC_TOPK = int(os.environ.get("RRR_PER_DOC_TOPK", "30"))
    MAX_SENTS_PER_PAGE = int(os.environ.get("RRR_MAX_SENTS_PAGE", "8"))
    MIN_CHARS = int(os.environ.get("RRR_MIN_SENT_CHARS", "20"))
    MIN_DOC_SNIPS = int(os.environ.get("RRR_MIN_DOC_SNIPS", "3"))
    GLOBAL_MIN_DOCS = int(os.environ.get("RRR_GLOBAL_MIN_DOCS", "5"))
    MD_QUOTE_CAP = int(os.environ.get("RRR_MD_QUOTE_CAP", "8"))
    DOC_BUDGET = int(os.environ.get("RRR_DOC_BUDGET", "24"))
    DOC_ADMIT_CACHE = os.environ.get("RRR_DOC_ADMIT_CACHE", "1") != "0"
    DOC_ADMIT_REPLAY = os.environ.get("RRR_DOC_ADMIT_REPLAY", "0") == "1"
    EV_CAP = int(os.environ.get("RRR_EV_PER_DOC_CAP", "8"))
    metrics.set("doc_admit_cache_enabled", int(DOC_ADMIT_CACHE))
    metrics.set("doc_admit_replay", int(DOC_ADMIT_REPLAY))

    from concurrent.futures import ThreadPoolExecutor, as_completed

    from rrr.query_planner import plan as plan_query
    with metrics.stage("planning"):
        plan_obj = plan_query(topic, metrics=metrics)
    score_query = " ".join(plan_obj.get("keywords_must", []) + plan_obj.get("keywords_any", []))
    score_query = score_query.strip() or topic
    probes = _plan_probes(plan_obj, topic, score_query)
    plan_obj["active_probes"] = probes
    metrics.set("planner_mode", plan_obj.get("planner_meta", {}).get("mode", "unknown"))
    metrics.set("planner_probe_count", len(probes))
    print(f"[Layered-T2] score_query={score_query}")
    print(f"[Layered-T2] probes={len(probes)}")

    admit_settings = {
        "per_doc_topk": PER_DOC_TOPK,
        "max_sents_per_page": MAX_SENTS_PER_PAGE,
        "min_chars": MIN_CHARS,
        "min_doc_snips": MIN_DOC_SNIPS,
        "ev_cap": EV_CAP,
        "min_sent_score": os.environ.get("RRR_MIN_SENT_SCORE", "40"),
        "bypass_validation": os.environ.get("RRR_BYPASS_VALIDATION", "0"),
    }
    admit_sig = _doc_admit_signature(topic, probes, all_doc_ids, admit_settings)
    write_run_manifest(
        "T2_LAYERED_GLOBAL",
        topic,
        meta_path,
        _MODEL,
        plan=plan_obj,
        extra={"admit_settings": admit_settings, "restart_attempt": restart_attempt},
    )

    MAX_WORKERS = int(os.environ.get("RRR_CONCURRENCY", "4"))
    admission_rejections = []
    rejection_lock = threading.Lock()

    def summarize_candidates(candidates):
        out = []
        for c in candidates[:10]:
            out.append({
                "page": int(c.get("page", 0) or 0),
                "bm25_score": round(float(c.get("bm25_score", 0.0) or 0.0), 4),
                "matched_probe_count": len(c.get("matched_probes", []) or []),
                "matched_probes": (c.get("matched_probes", []) or [])[:4],
            })
        return out

    def record_rejection(did, reason, candidates=None, **details):
        item = {
            "doc_id": did,
            "citation": refs.get(did, did),
            "reason": reason,
            "thresholds": {
                "min_doc_snips": MIN_DOC_SNIPS,
                "min_chars": MIN_CHARS,
                "min_sent_score": admit_settings["min_sent_score"],
                "per_doc_topk": PER_DOC_TOPK,
                "max_sents_per_page": MAX_SENTS_PER_PAGE,
            },
            "candidate_pages": summarize_candidates(candidates or []),
        }
        item.update(details)
        with rejection_lock:
            admission_rejections.append(item)

    print(f"[Layered-T2] starting evidence admission over {len(all_doc_ids)} docs (concurrency={MAX_WORKERS})")

    def process_doc(did):
        metrics.inc("docs_processed")
        candidates = _retrieve_doc_with_probes(retrieve, did, probes, PER_DOC_TOPK, metrics=metrics)
        metrics.inc("retrieved_pages_kept", len(candidates))
        if not candidates:
            record_rejection(did, "no_retrieved_pages", candidates=[])
            metrics.inc("docs_rejected_no_pages")
            return None

        quotes = []
        page_sentence_counts = {}
        for c in candidates:
            txt = c.get("text", "").strip()
            if not txt:
                page_sentence_counts[int(c.get("page", 0) or 0)] = 0
                continue
            scored_sents = select_sentences(
                txt,
                topic,
                max_sentences=MAX_SENTS_PER_PAGE,
                min_chars=MIN_CHARS,
                probes=probes,
            )
            page_sentence_counts[int(c.get("page", 0) or 0)] = len(scored_sents)
            for sent, score in scored_sents:
                s_norm = normalize_space(sent)
                if len(s_norm) < MIN_CHARS:
                    continue
                best_probe = _best_probe_for_sentence(s_norm, c.get("matched_probes") or probes)
                quotes.append({
                    "type": "quote",
                    "doc_id": did,
                    "page": int(c["page"]),
                    "text": s_norm,
                    "score": score,
                    "best_probe": best_probe,
                    "matched_probes": c.get("matched_probes", []),
                })
        metrics.inc("candidate_quotes", len(quotes))

        seen = {}
        for q in quotes:
            k = (q["page"], q["text"][:160])
            if k not in seen or q["score"] > seen[k]["score"]:
                seen[k] = q
        quotes = list(seen.values())

        if len(quotes) < MIN_DOC_SNIPS:
            metrics.inc("docs_rejected_min_quotes")
            record_rejection(
                did,
                "insufficient_candidate_sentences",
                candidates=candidates,
                selected_sentence_count=len(quotes),
                page_sentence_counts=page_sentence_counts,
            )
            return None

        verbose_val = validate_evidence_verbose(quotes, df)
        val = [{"item": v["item"], "ok": v["verdict"] in ("exact", "soft_ok", "bypass"), "reason": v["reason"]} for v in verbose_val]
        metrics.inc("validation_items", len(val))
        metrics.inc("validation_ok", sum(1 for v in val if v["ok"]))
        metrics.inc("validation_failed", sum(1 for v in val if not v["ok"]))
        valid_quotes = [v["item"] for v in val if v["ok"]]
        if len(valid_quotes) < MIN_DOC_SNIPS:
            metrics.inc("docs_rejected_validation")
            reason_counts = {}
            for v in verbose_val:
                reason = v.get("reason") or v.get("verdict") or "unknown"
                if v.get("verdict") not in ("exact", "soft_ok", "bypass"):
                    reason_counts[reason] = reason_counts.get(reason, 0) + 1
            record_rejection(
                did,
                "insufficient_validated_quotes",
                candidates=candidates,
                selected_sentence_count=len(quotes),
                validation_items=len(val),
                validation_ok=len(valid_quotes),
                validation_failed=len(val) - len(valid_quotes),
                validation_reason_counts=reason_counts,
            )
            return None

        valid_quotes = _rerank_quotes_for_diversity(valid_quotes, EV_CAP)
        metrics.inc("valid_quotes_kept", len(valid_quotes))

        avg_score = sum(q.get("score", 0) for q in valid_quotes) / len(valid_quotes) if valid_quotes else 0
        probe_hits = []
        for q in valid_quotes:
            for probe in q.get("matched_probes", []) or [q.get("best_probe")]:
                if probe and probe not in probe_hits:
                    probe_hits.append(probe)

        metrics.inc("docs_admitted")
        return {
            "doc_id": did,
            "citation": refs.get(did, did),
            "quotes": valid_quotes,
            "avg_score": round(avg_score, 2),
            "probe_hits": probe_hits,
        }

    def enrich_doc(doc):
        did = doc["doc_id"]
        valid_quotes = doc.get("quotes", [])
        ev_texts = []
        for q in valid_quotes:
            text = q.get("text", "")
            clipped = (text[:220] + "...") if len(text) > 220 else text
            ev_texts.append(f"[{q['doc_id']} p.{q['page']}]\n- {clipped}")

        evidence_stance = classify_evidence_stance(did, topic, valid_quotes, metrics=metrics)

        topic_words = set(w.lower() for w in re.findall(r'\b\w{4,}\b', topic))
        topic_words_str = ", ".join(sorted(topic_words))
        mechanism_prompt = (
            "Extract the key causal mechanisms from this document using ONLY the quotes provided.\n\n"
            f"Topic:\n{topic}\n\n"
            "Quotes:\n" + "\n\n---\n\n".join(ev_texts) + "\n\n"
            "Return ONE JSON object:\n"
            "{\n"
            '  "mechanisms": ["specific mechanism 1", "specific mechanism 2"]\n'
            "}\n\n"
            "Rules for mechanisms:\n"
            "- Each mechanism must answer HOW or THROUGH WHAT\n"
            "- Must contain at least one concrete noun NOT in this list: [" + topic_words_str + "]\n"
            "- Name a specific causal pathway, variable, condition, or empirical referent\n"
            "- 8-15 words per mechanism\n"
            "- Maximum 3 mechanisms per document\n"
        )

        sig = _mechanism_signature(topic, valid_quotes)
        cached = _load_mechanism_cache(did, sig)
        if cached and cached.get("mechanisms"):
            metrics.cache_event("mechanism", "hits")
            mechanisms = cached.get("mechanisms", [])
        else:
            metrics.cache_event("mechanism", "misses")
            try:
                import ollama
                start = time.perf_counter()
                res = ollama.chat(
                    model=_MODEL,
                    messages=[{"role": "user", "content": mechanism_prompt}],
                    options=_OPTIONS_MECHANISM, keep_alive=_KEEP_ALIVE, stream=False
                )
                mech_raw = res["message"]["content"].strip()
                metrics.record_llm("mechanism", _MODEL, options=_OPTIONS_MECHANISM,
                                   duration_s=time.perf_counter() - start,
                                   prompt_chars=len(mechanism_prompt),
                                   response_chars=len(mech_raw))
                try:
                    mech_obj = json.loads(mech_raw[mech_raw.find("{"):])
                    mechanisms = mech_obj.get("mechanisms", [])
                except Exception:
                    mechanisms = []
            except Exception as e:
                mechanisms = []
                metrics.record_llm("mechanism", _MODEL, options=_OPTIONS_MECHANISM,
                                   success=False, error=e)
            if mechanisms:
                _save_mechanism_cache(
                    did,
                    sig,
                    {
                        "mechanisms": mechanisms,
                        "model": _MODEL,
                        "prompt_version": _MECHANISM_PROMPT_VERSION,
                    },
                )
                metrics.cache_event("mechanism", "writes")
            else:
                metrics.cache_event("mechanism", "skips")
        metrics.inc("mechanisms_extracted", len(mechanisms))
        metrics.inc("docs_kept")

        enriched = dict(doc)
        enriched["stance"] = evidence_stance
        enriched["mechanisms"] = mechanisms
        return enriched

    admitted_docs = None
    if DOC_ADMIT_CACHE and DOC_ADMIT_REPLAY:
        cached_admit = _load_doc_admit_cache_obj(admit_sig)
        if cached_admit is not None:
            admitted_docs = cached_admit.get("docs", [])
            admission_rejections.extend(cached_admit.get("rejections", []) or [])
            metrics.cache_event("doc_admit", "hits")
            metrics.set("doc_admit_cache_key", admit_sig)
        else:
            metrics.cache_event("doc_admit", "misses")
    elif DOC_ADMIT_CACHE:
        metrics.cache_event("doc_admit", "skips")
        metrics.set("doc_admit_cache_key", admit_sig)

    if admitted_docs is None:
        admitted_docs = []
        with metrics.stage("evidence_admission"):
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
                futures = {pool.submit(process_doc, did): did for did in all_doc_ids}
                for fut in as_completed(futures):
                    res = fut.result()
                    if res:
                        admitted_docs.append(res)
        admitted_docs = sorted(admitted_docs, key=lambda x: x.get("avg_score", 0), reverse=True)
        admission_rejections = sorted(admission_rejections, key=lambda x: x.get("doc_id", ""))
        if DOC_ADMIT_CACHE:
            _save_doc_admit_cache(admit_sig, admitted_docs, admit_settings, rejections=admission_rejections)
            metrics.cache_event("doc_admit", "writes")
            metrics.set("doc_admit_cache_key", admit_sig)
    _write_json_run("admission_rejections.json", {
        "topic": topic,
        "cache_key": admit_sig,
        "rejections": admission_rejections,
    })
    metrics.set("docs_rejected_total", len(admission_rejections))

    metrics.set("docs_admitted_total", len(admitted_docs))
    selected_docs = _select_budget_docs(admitted_docs, DOC_BUDGET, probes, metrics=metrics)
    metrics.set("docs_selected_for_llm", len(selected_docs))
    topic_fit = _compute_topic_fit(
        topic,
        probes,
        all_doc_ids,
        admitted_docs,
        selected_docs,
        rejections=admission_rejections,
    )
    _write_json_run("topic_fit.json", topic_fit)
    metrics.set("topic_fit_warnings", topic_fit.get("warnings", []))
    metrics.set("topic_fit_admission_share", topic_fit.get("admission_share"))
    metrics.set("topic_fit_selected_probe_coverage", topic_fit.get("selected_probe_coverage"))
    print(f"[Layered-T2] admitted={len(admitted_docs)} selected_for_llm={len(selected_docs)} budget={DOC_BUDGET}")

    doc_summaries = []
    with metrics.stage("per_document_sweep"):
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {pool.submit(enrich_doc, doc): doc.get("doc_id") for doc in selected_docs}
            for fut in as_completed(futures):
                res = fut.result()
                if res:
                    doc_summaries.append(res)

    print("[Layered-T2] clustering mechanisms...")
    with metrics.stage("clustering"):
        mech_to_cluster = _cluster_mechanisms(doc_summaries, topic, metrics=metrics)
    
    for doc in doc_summaries:
        primary = None
        for m in doc.get("mechanisms", []):
            m = (m or "").strip()
            if m:
                primary = m
                break
        doc["cluster"] = mech_to_cluster.get(primary, "Other") if primary else "Other"

    evidence_id_count = _assign_evidence_ids(doc_summaries)
    metrics.set("evidence_ids_assigned", evidence_id_count)

    for entry in doc_summaries:
        did = entry["doc_id"]
        with open(runs_path("layered_docs", f"{did}.json"), "w", encoding="utf-8") as f:
            json.dump(entry, f, indent=2, ensure_ascii=False)

    kept = len(doc_summaries)
    metrics.set("docs_represented", kept)
    topic_fit = _compute_topic_fit(
        topic,
        probes,
        all_doc_ids,
        admitted_docs,
        selected_docs,
        doc_summaries=doc_summaries,
        rejections=admission_rejections,
    )
    _write_json_run("topic_fit.json", topic_fit)
    metrics.set("topic_fit_warnings", topic_fit.get("warnings", []))
    metrics.set("topic_fit_represented_share", topic_fit.get("represented_share"))
    print(f"[Layered-T2] per-document sweep complete: {kept} docs summarised")

    # Print stance distribution
    from collections import Counter
    stance_counts = Counter(d.get("stance", "tangential") for d in doc_summaries)
    metrics.set("stance_distribution", dict(stance_counts))
    print(f"[Layered-T2] stance distribution: {dict(stance_counts)}")

    if kept < GLOBAL_MIN_DOCS:
        print("[Layered-T2] refusal=insufficient_global_evidence")
        write_run("T2_LAYERED_GLOBAL", topic, {"docs_seen": len(all_doc_ids), "docs_represented": kept},
                  {"refusal": True, "reason": "insufficient_global_evidence"})
        metrics.set("refusal", True)
        metrics.set("refusal_reason", "insufficient_global_evidence")
        write_run_manifest(
            "T2_LAYERED_GLOBAL",
            topic,
            meta_path,
            _MODEL,
            plan=plan_obj,
            extra={"admit_settings": admit_settings, "topic_fit": topic_fit, "refusal": "insufficient_global_evidence"},
        )
        metrics.save()
        return

    import collections
    def _render_review_narrative(topic, doc_summaries, meta_n_total):
        def norm(s): return re.sub(r"\s+"," ",(s or "").strip())
        counts = collections.Counter(x.get("stance","tangential") for x in doc_summaries)
        
        cluster_counts = collections.Counter(x.get("cluster", "Other") for x in doc_summaries)
        
        def top_mechs(stance,k=8):
            c=collections.Counter()
            for x in doc_summaries:
                if x.get("stance")==stance:
                    for m in x.get("mechanisms",[]):
                        m=norm(m)
                        if m: c[m]+=1
            return [m for m,_ in c.most_common(k)]
        def notable(stance,k=6):
            cand=[(x.get("citation") or x["doc_id"], x.get("avg_score", 0))
                  for x in doc_summaries if x.get("stance")==stance]
            cand.sort(key=lambda t:t[1], reverse=True)
            return cand[:k]

        lines=[]
        lines.append("# Literature review\n")
        lines.append(f"**Topic:** {topic}\n")
        lines.append(f"**Coverage:** {len(doc_summaries)} of {meta_n_total} documents.\n")
        lines.append("**Stance distribution:** " + ", ".join(
            f"{k}: {counts.get(k,0)}" for k in ["supports","critiques","complicates","tangential"]
        ) + "\n")
        lines.append("**Thematic clusters:** " + ", ".join(
            f"{k} ({v})" for k, v in cluster_counts.most_common()
        ) + "\n")
        lines.append("---\n")
        for sec in ["supports","critiques","complicates"]:
            if counts.get(sec,0)==0: continue
            lines.append(f"## {sec.capitalize()}\n")
            mechs = top_mechs(sec, 8)
            if mechs:
                lines.append("**Common mechanisms/themes:**")
                lines += [f"- {m}" for m in mechs] + [""]
            nd = notable(sec, 6)
            if nd:
                lines.append("**Notable documents (by evidence relevance):**")
                lines += [f"- {c} - avg score {s:.1f}" for c, s in nd] + [""]

        return "\n".join(lines)

    ensure_dir(str(runs_path()))
    
    ledger_data = {
        "topic": topic,
        "plan": plan_obj,
        "bypass_condition": os.environ.get("RRR_BYPASS_VALIDATION", "0") == "1",
        "topic_fit": topic_fit,
        "admission": {
            "cache_key": admit_sig,
            "settings": admit_settings,
            "docs_admitted": len(admitted_docs),
            "docs_rejected": len(admission_rejections),
            "docs_selected_for_llm": len(selected_docs),
        },
        "docs": doc_summaries,
        "restarts_required": restart_attempt
    }
    with metrics.stage("write_ledger"):
        with open(runs_path("review_ledger.json"), "w", encoding="utf-8") as f:
            json.dump(ledger_data, f, indent=2, ensure_ascii=False)
        with open(runs_path("plan.json"), "w", encoding="utf-8") as f:
            json.dump(plan_obj, f, indent=2, ensure_ascii=False)

    narrative_md = _render_review_narrative(topic, doc_summaries, len(all_doc_ids))
    with open(runs_path("review_narrative.md"), "w", encoding="utf-8") as f:
        f.write(narrative_md)

    if not getattr(args, "narrative_only", False):
        md_lines = [f"# Literature Review\n", f"**Topic:** {topic}\n", f"\n**Coverage:** {kept} of {len(all_doc_ids)} documents.\n", "\n---\n"]
        def stance_key(s):
            return {"supports":0, "complicates":1, "critiques":2, "tangential":3}.get(s.get("stance","tangential"), 3)
        for entry in sorted(doc_summaries, key=stance_key):
            md_lines.append(f"## {entry['citation']}")
            md_lines.append(f"**Stance:** {entry['stance']} | **Cluster:** {entry.get('cluster', 'Other')} | **Relevance:** {entry.get('avg_score', 0):.1f}")
            if entry["mechanisms"]:
                md_lines.append("**Mechanisms:**"); [md_lines.append(f"- {m}") for m in entry["mechanisms"]]
            if entry["quotes"]:
                md_lines.append("**Quotes (page-level, with scores):**")
                for q in entry["quotes"][:MD_QUOTE_CAP]:
                    md_lines.append(f"- p.{q['page']} [score={q.get('score',0):.0f}]: \"{q['text']}\"")
            md_lines.append("")
        with open(runs_path("T2_review.md"), "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines))

    if os.environ.get("RRR_WRITE_REVIEW", "0") != "1":
        print("\n[Review narrative]\n")
        print(narrative_md)
    print("\n[Layered-T2] wrote: runs/review_narrative.md and runs/review_ledger.json")
    if not getattr(args, "narrative_only", False):
        print("[Layered-T2] appendix: runs/T2_review.md")

    if os.environ.get("RRR_WRITE_REVIEW", "0") == "1":
        from rrr.writer import compose_from_ledger
        print("[Layered-T2] composing long-form literature review...")
        try:
            with metrics.stage("writing"):
                composed_path = compose_from_ledger(str(runs_path("review_ledger.json")), metrics=metrics)
            print(f"[Layered-T2] composed review written at: {composed_path}")

            with open(composed_path, "r", encoding="utf-8") as f:
                long_form = f.read()

            print("\n" + "="*80)
            print("LONG-FORM LITERATURE REVIEW")
            print("="*80 + "\n")
            print(long_form)
            print("\n" + "="*80 + "\n")

            allowed_docs = set()
            allowed_pages_by_doc = {}
            for d in doc_summaries:
                did = str(d.get("doc_id", "")).strip()
                if not did:
                    continue
                allowed_docs.add(did)
                allowed_pages_by_doc.setdefault(did, set())
                for q in (d.get("quotes") or []):
                    qdid = str(q.get("doc_id", did)).strip() or did
                    try:
                        pg = int(q.get("page", 0) or 0)
                    except Exception:
                        pg = 0
                    if qdid and pg > 0:
                        allowed_docs.add(qdid)
                        allowed_pages_by_doc.setdefault(qdid, set()).add(pg)

            cited_docs_path = str(runs_path("review_cited_docs.json"))
            if os.path.isfile(cited_docs_path):
                with open(cited_docs_path, "r", encoding="utf-8") as f:
                    cited_docids = json.load(f)
            else:
                author_year_to_docid = _build_author_year_lookup(allowed_docs)
                cited_docs = _collect_cited_docs(long_form, allowed_docs, author_year_to_docid)
                cited_docids = list(cited_docs)

            if not cited_docids:
                ensure_dir(str(runs_path()))
                with open(runs_path("review_reference_build.failures.txt"), "w", encoding="utf-8") as f:
                    f.write("No valid citations found in review_composed.md.\n")
                print("\n" + "="*80)
                print("REFERENCES (cited in review)")
                print("="*80 + "\n")
                print("[REFUSAL] No citations found. See runs/review_reference_build.failures.txt")
                print("\n" + "="*80 + "\n")
                metrics.set("refusal", True)
                metrics.set("refusal_reason", "no_citations_found")
                write_run_manifest(
                    "T2_LAYERED_GLOBAL",
                    topic,
                    meta_path,
                    _MODEL,
                    plan=plan_obj,
                    extra={"admit_settings": admit_settings, "topic_fit": topic_fit, "refusal": "no_citations_found"},
                )
                metrics.save()
                return

            raw_ref_lines = [(did, refs.get(did, did)) for did in cited_docids]
            
            def sort_key(item):
                did = item[0]
                clean = did.replace("EtAl", "").replace("&", "")
                parts = clean.split("_")
                return parts[0].lower() if parts else did.lower()
            
            raw_ref_lines = sorted(raw_ref_lines, key=sort_key)
            
            seen_refs = set()
            ref_lines = []
            for did, rline in raw_ref_lines:
                if rline not in seen_refs:
                    ref_lines.append(rline)
                    seen_refs.add(rline)

            print("\n" + "="*80)
            print("REFERENCES (cited in review)")
            print("="*80 + "\n")
            for i, rline in enumerate(ref_lines, start=1):
                print(f"{i}. {rline}")
            print("\n" + "="*80 + "\n")

            ensure_dir(str(runs_path()))
            with open(runs_path("review_references.txt"), "w", encoding="utf-8") as f:
                for i, rline in enumerate(ref_lines, start=1):
                    f.write(f"{i}. {rline}\n")

        except Exception as e:
            print(f"[Layered-T2] writer failed: {e}")
            metrics.set("writer_error", str(e))

    metrics.set("refusal", False)
    write_run_manifest(
        "T2_LAYERED_GLOBAL",
        topic,
        meta_path,
        _MODEL,
        plan=plan_obj,
        extra={
            "admit_settings": admit_settings,
            "topic_fit": topic_fit,
            "outputs": [
                "review_ledger.json",
                "review_narrative.md",
                "topic_fit.json",
                "admission_rejections.json",
                "run_metrics.json",
            ],
        },
    )
    metrics.save()


def layered_t2(args, meta_path):
    """
    Main entry point for layered T2 with automatic restart on clustering failure.
    """
    MAX_RESTARTS = 5
    
    for restart_attempt in range(MAX_RESTARTS):
        try:
            if restart_attempt > 0:
                print(f"[Layered-T2] === RESTART {restart_attempt}/{MAX_RESTARTS} ===")
            
            _layered_t2_inner(args, meta_path, restart_attempt)
            return
            
        except ClusteringFailedError as e:
            print(f"[Layered-T2] {e}")
            if restart_attempt < MAX_RESTARTS - 1:
                print(f"[Layered-T2] Restarting full pipeline...")
                import shutil
                cache_path = str(runs_path("cache", "mechanisms"))
                if os.path.isdir(cache_path):
                    shutil.rmtree(cache_path)
                continue
            else:
                raise RuntimeError(f"Pipeline failed after {MAX_RESTARTS} full restarts due to clustering failures")

