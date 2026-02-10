from typing import List
import os, subprocess, json, re, ast, hashlib
from rrr.utils import ensure_dir
from rrr.stance import classify_stance

_MODEL      = os.environ.get("RRR_MODEL", "mistral")
_OPTIONS    = {"temperature": 0.0, "num_ctx": 32768}
_KEEP_ALIVE = "30m"


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
        "After printing the JSON, output the word SUMMARY on a new line and then write your 2–6-sentence summary.\n"
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
            os.makedirs("logs", exist_ok=True)
            with open("logs/invalid_json.txt", "w", encoding="utf-8") as f:
                f.write(raw)
            return None, raw.strip()

def reason_over_evidence(evidence_texts: List[str], claim: str, model: str = _MODEL) -> str:
    if not evidence_texts:
        return "No evidence to reason over."
    prompt = _build_prompt(evidence_texts, claim)
    try:
        import ollama
        res = ollama.chat(model=model,
                          messages=[{"role":"user","content":prompt}],
                          options=_OPTIONS, keep_alive=_KEEP_ALIVE, stream=False)
        out = res["message"]["content"].strip()
    except Exception as e_client:
        try:
            p = subprocess.run(["ollama","run", model],
                               input=prompt.encode("utf-8"),
                               capture_output=True, timeout=600)
            out = p.stdout.decode("utf-8").strip() or "(no output)"
        except Exception as e_sub:
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

def _mechanism_signature(topic: str, quotes) -> str:
    h = hashlib.sha256()
    h.update((topic or "").encode("utf-8"))
    for q in quotes:
        h.update(f"{q.get('doc_id','')}|{q.get('page','')}|{q.get('text','')}\n".encode("utf-8"))
    return h.hexdigest()[:16]

def _mechanism_cache_path(doc_id: str, sig: str) -> str:
    return os.path.join("runs", "cache", "mechanisms", f"{doc_id}_{sig}.json")

def _load_mechanism_cache(doc_id: str, sig: str):
    try:
        with open(_mechanism_cache_path(doc_id, sig), encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def _save_mechanism_cache(doc_id: str, sig: str, obj):
    ensure_dir(os.path.join("runs", "cache", "mechanisms"))
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

def _cluster_mechanisms(doc_summaries: list, topic: str) -> dict:
    """
    Cluster mechanisms into themes.
    
    Raises ClusteringFailedError after MAX_RETRIES failures - caller should restart pipeline.
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
        return {}
    
    all_mechs = list(mech_to_docs.keys())
    n_mechs = len(all_mechs)
    
    if n_mechs <= 3:
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
            res = ollama.chat(
                model=_MODEL,
                messages=[{"role": "user", "content": cluster_prompt}],
                options={"temperature": 0.0, "num_ctx": 16384, "num_predict": 6000},
                keep_alive=_KEEP_ALIVE,
                stream=False
            )
            raw = res["message"]["content"].strip()
            
            ensure_dir("logs")
            with open(f"logs/cluster_raw_attempt{attempt+1}.txt", "w", encoding="utf-8") as f:
                f.write(raw)
            
            mech_to_cluster = _try_parse_cluster_json(raw, n_mechs, all_mechs)
            
            if mech_to_cluster is not None:
                break
            
            print(f"[Clustering] Attempt {attempt+1}/{MAX_RETRIES} failed to parse JSON, retrying...")
            
        except Exception as e:
            print(f"[Clustering] Attempt {attempt+1}/{MAX_RETRIES} error: {e}")
            continue
    
    if mech_to_cluster is None:
        raise ClusteringFailedError(f"Clustering failed after {MAX_RETRIES} attempts")
    
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
        (r"\\'e", 'é'), (r"\\`e", 'è'), (r'\\"e', 'ë'), (r'\\^e', 'ê'),
        (r"\\'a", 'á'), (r"\\`a", 'à'), (r'\\"a', 'ä'), (r'\\^a', 'â'),
        (r"\\'o", 'ó'), (r"\\`o", 'ò'), (r'\\"o', 'ö'), (r'\\^o', 'ô'),
        (r"\\'u", 'ú'), (r"\\`u", 'ù'), (r'\\"u', 'ü'), (r'\\^u', 'û'),
        (r"\\'i", 'í'), (r"\\`i", 'ì'), (r'\\"i', 'ï'), (r'\\^i', 'î'),
        (r'\\c{c}', 'ç'), (r'\\c{C}', 'Ç'),
        (r'\\c{s}', 'ş'), (r'\\c{S}', 'Ş'),
        (r'\\v{s}', 'š'), (r'\\v{S}', 'Š'),
        (r'\\~n', 'ñ'), (r'\\~N', 'Ñ'),
        (r'\\ss', 'ß'),
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


def _layered_t2_inner(args, meta_path, restart_attempt=0):
    """
    Inner implementation of layered_t2.
    """
    import os, json
    import pandas as pd
    from rrr.retrieve import retrieve
    from rrr.evidence_filter import select_sentences
    from rrr.validate import validate_evidence
    from rrr.utils import ensure_dir, write_run, normalize_space

    topic = args.topic
    df = pd.read_csv(meta_path)
    df["doc_id"] = df["doc_id"].astype(str)

    refs = {str(r["doc_id"]): _cite_harvard(r) for _, r in df.iterrows()}
    all_doc_ids = df["doc_id"].tolist()

    ensure_dir("runs"); ensure_dir("runs/layered_docs")

    PER_DOC_TOPK = int(os.environ.get("RRR_PER_DOC_TOPK", "30"))
    MAX_SENTS_PER_PAGE = int(os.environ.get("RRR_MAX_SENTS_PAGE", "8"))
    MIN_CHARS = int(os.environ.get("RRR_MIN_SENT_CHARS", "20"))
    MIN_DOC_SNIPS = int(os.environ.get("RRR_MIN_DOC_SNIPS", "3"))
    GLOBAL_MIN_DOCS = int(os.environ.get("RRR_GLOBAL_MIN_DOCS", "5"))
    MD_QUOTE_CAP = int(os.environ.get("RRR_MD_QUOTE_CAP", "8"))

    from concurrent.futures import ThreadPoolExecutor, as_completed

    from rrr.query_planner import plan as plan_query
    plan_obj = plan_query(topic)
    score_query = " ".join(plan_obj.get("keywords_must", []) + plan_obj.get("keywords_any", []))
    score_query = score_query.strip() or topic
    print(f"[Layered-T2] score_query={score_query}")

    MAX_WORKERS = int(os.environ.get("RRR_CONCURRENCY", "4"))

    print(f"[Layered-T2] starting per-document sweep over {len(all_doc_ids)} docs (concurrency={MAX_WORKERS})")

    def process_doc(did):
        candidates = retrieve(score_query, topk=PER_DOC_TOPK, doc_id=did)

        quotes = []
        for c in candidates:
            txt = c.get("text", "").strip()
            if not txt:
                continue
            scored_sents = select_sentences(txt, topic, max_sentences=MAX_SENTS_PER_PAGE, min_chars=MIN_CHARS)
            for sent, score in scored_sents:
                s_norm = normalize_space(sent)
                if len(s_norm) < MIN_CHARS:
                    continue
                quotes.append({
                    "type": "quote",
                    "doc_id": did,
                    "page": int(c["page"]),
                    "text": s_norm,
                    "score": score
                })

        seen = {}
        for q in quotes:
            k = (q["page"], q["text"][:160])
            if k not in seen or q["score"] > seen[k]["score"]:
                seen[k] = q
        quotes = list(seen.values())

        if len(quotes) < MIN_DOC_SNIPS:
            return None

        val = validate_evidence(quotes, df)
        valid_quotes = [v["item"] for v in val if v["ok"]]
        if len(valid_quotes) < MIN_DOC_SNIPS:
            return None

        valid_quotes = sorted(valid_quotes, key=lambda x: x.get("score", 0), reverse=True)
        EV_CAP = int(os.environ.get("RRR_EV_PER_DOC_CAP", "8"))
        valid_quotes = valid_quotes[:EV_CAP]

        def _clip(s, n=220):
            return (s[:n] + "…") if len(s) > n else s
        ev_texts = [f"[{q['doc_id']} p.{q['page']}]\n- {_clip(q['text'])}" for q in valid_quotes]

        # ============================================================
        # STANCE: Classify from abstract (cached by doc_id + topic)
        # ============================================================
        abstract_stance = classify_stance(did, topic)

        # ============================================================
        # MECHANISMS: Extract from snippets (unchanged logic)
        # ============================================================
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
        if cached and "mechanisms" in cached:
            mechanisms = cached.get("mechanisms", [])
        else:
            try:
                import ollama
                res = ollama.chat(
                    model=_MODEL,
                    messages=[{"role": "user", "content": mechanism_prompt}],
                    options=_OPTIONS, keep_alive=_KEEP_ALIVE, stream=False
                )
                mech_raw = res["message"]["content"].strip()
                try:
                    mech_obj = json.loads(mech_raw[mech_raw.find("{"):])
                    mechanisms = mech_obj.get("mechanisms", [])
                except Exception:
                    mechanisms = []
            except Exception:
                mechanisms = []
            _save_mechanism_cache(did, sig, {"mechanisms": mechanisms})

        avg_score = sum(q.get("score", 0) for q in valid_quotes) / len(valid_quotes) if valid_quotes else 0

        return {
            "doc_id": did,
            "citation": refs.get(did, did),
            "stance": abstract_stance,  # FROM ABSTRACT
            "mechanisms": mechanisms,
            "quotes": valid_quotes,
            "avg_score": round(avg_score, 2)
        }

    doc_summaries = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(process_doc, did): did for did in all_doc_ids}
        for fut in as_completed(futures):
            res = fut.result()
            if res:
                doc_summaries.append(res)

    print("[Layered-T2] clustering mechanisms...")
    mech_to_cluster = _cluster_mechanisms(doc_summaries, topic)
    
    for doc in doc_summaries:
        primary = None
        for m in doc.get("mechanisms", []):
            m = (m or "").strip()
            if m:
                primary = m
                break
        doc["cluster"] = mech_to_cluster.get(primary, "Other") if primary else "Other"

    for entry in doc_summaries:
        did = entry["doc_id"]
        with open(os.path.join("runs", "layered_docs", f"{did}.json"), "w", encoding="utf-8") as f:
            json.dump(entry, f, indent=2, ensure_ascii=False)

    kept = len(doc_summaries)
    print(f"[Layered-T2] per-document sweep complete: {kept} docs summarised")

    # Print stance distribution
    from collections import Counter
    stance_counts = Counter(d.get("stance", "tangential") for d in doc_summaries)
    print(f"[Layered-T2] stance distribution: {dict(stance_counts)}")

    if kept < GLOBAL_MIN_DOCS:
        print("[Layered-T2] refusal=insufficient_global_evidence")
        write_run("T2_LAYERED_GLOBAL", topic, {"docs_seen": len(all_doc_ids), "docs_represented": kept},
                  {"refusal": True, "reason": "insufficient_global_evidence"})
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

    ensure_dir("runs")
    
    ledger_data = {
        "topic": topic,
        "docs": doc_summaries,
        "restarts_required": restart_attempt
    }
    with open(os.path.join("runs","review_ledger.json"), "w", encoding="utf-8") as f:
        json.dump(ledger_data, f, indent=2, ensure_ascii=False)

    narrative_md = _render_review_narrative(topic, doc_summaries, len(all_doc_ids))
    with open(os.path.join("runs","review_narrative.md"), "w", encoding="utf-8") as f:
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
        with open(os.path.join("runs","T2_review.md"), "w", encoding="utf-8") as f:
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
            composed_path = compose_from_ledger(os.path.join("runs", "review_ledger.json"))
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

            cited_docs_path = os.path.join("runs", "review_cited_docs.json")
            if os.path.isfile(cited_docs_path):
                with open(cited_docs_path, "r", encoding="utf-8") as f:
                    cited_docids = json.load(f)
            else:
                author_year_to_docid = _build_author_year_lookup(allowed_docs)
                cited_docs = _collect_cited_docs(long_form, allowed_docs, author_year_to_docid)
                cited_docids = list(cited_docs)

            if not cited_docids:
                ensure_dir("runs")
                with open(os.path.join("runs", "review_reference_build.failures.txt"), "w", encoding="utf-8") as f:
                    f.write("No valid citations found in review_composed.md.\n")
                print("\n" + "="*80)
                print("REFERENCES (cited in review)")
                print("="*80 + "\n")
                print("[REFUSAL] No citations found. See runs/review_reference_build.failures.txt")
                print("\n" + "="*80 + "\n")
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

            ensure_dir("runs")
            with open(os.path.join("runs", "review_references.txt"), "w", encoding="utf-8") as f:
                for i, rline in enumerate(ref_lines, start=1):
                    f.write(f"{i}. {rline}\n")

        except Exception as e:
            print(f"[Layered-T2] writer failed: {e}")


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
                cache_path = os.path.join("runs", "cache", "mechanisms")
                if os.path.isdir(cache_path):
                    shutil.rmtree(cache_path)
                continue
            else:
                raise RuntimeError(f"Pipeline failed after {MAX_RESTARTS} full restarts due to clustering failures")

