import argparse, uuid, pandas as pd, os, json
from rrr.retrieve import retrieve
from rrr.reasoner import reason_over_evidence, layered_t2
from rrr.utils import write_run
from rrr.evidence_filter import select_sentences
from rrr.validate import validate_evidence

def load_refs(meta_path):
    df = pd.read_csv(meta_path)
    refs = {}
    for _, r in df.iterrows():
        def safe(x): return str(x).strip() if isinstance(x, str) else ""
        y = f" ({safe(r.get('year'))})" if 'year' in r and pd.notna(r.get('year')) and safe(r.get('year')) else ""
        # robust fallback
        parts = []
        if safe(r.get('authors')): parts.append(safe(r.get('authors')) + y)
        if safe(r.get('title')):   parts.append(f"*{safe(r.get('title'))}*")
        if safe(r.get('venue')):   parts.append(safe(r.get('venue')))
        cite = ". ".join(parts).strip() or str(r.get('doc_id'))
        refs[str(r["doc_id"])] = cite
    return df, refs

def render_markdown(obj, refs):
    lines = [f"### Claim / Topic\n{obj.get('claim') or obj.get('topic','')}\n"]
    for ev in obj.get('evidence', []):
        ref = refs.get(ev['doc_id'], ev['doc_id'])
        lines.append(f"- p.{ev['page']} - {ref}")
        snippet = ev['text'][:350].replace("\n"," ")
        lines.append(f"  > {snippet}...\n")
    return "\n".join(lines)

def t1(args, meta_path):
    df, refs = load_refs(meta_path)
    topk = int(os.environ.get('RRR_TOPK', '12'))
    MIN_DOCS = int(os.environ.get("RRR_MIN_DOCS", "5"))
    MIN_SNIPS = int(os.environ.get("RRR_MIN_SNIPS", "20"))
    candidates = retrieve(args.claim, topk=topk)
    filtered = []
    for c in candidates:
        if not c["text"].strip(): continue
        sents = select_sentences(c["text"], args.claim, max_sentences=6)
        if not sents: continue
        filtered.append({"doc_id": c["doc_id"], "page": c["page"], "text": " ".join(sents)})
    distinct_docs = {e["doc_id"] for e in filtered}
    total_snips = sum(ev["text"].count(". ") + 1 for ev in filtered)
    if len(distinct_docs) < MIN_DOCS or total_snips < MIN_SNIPS:
        print("[T1] refusal=insufficient_evidence_floor")
        write_run("T1", args.claim, candidates, {"refusal": True, "reason": "insufficient_evidence_floor"})
        return
    evidence_texts = []
    for e in filtered:
        header = f"[{e['doc_id']} p.{e['page']}]"
        evidence_texts.append(header + "\n- " + e["text"])
    reasoned = reason_over_evidence(evidence_texts, args.claim)
    run_id = uuid.uuid4().hex[:8]
    write_run("T1", args.claim, filtered, {"refusal": False, "reasoned": reasoned})
    print("\n[Reasoning output]\n", reasoned)
    print(f"[T1] run_id={run_id} refusal=False")

def t2(args, meta_path):
    if args.multi:
        layered_t2(args, meta_path)  # args carries narrative_only; layered_t2 will print narrative
        return

    # Strict, single-pass path (kept unchanged)
    df, refs = load_refs(meta_path)
    topk = int(os.environ.get('RRR_TOPK', '12'))
    MIN_DOCS = int(os.environ.get("RRR_MIN_DOCS", "6"))
    MIN_SNIPS = int(os.environ.get("RRR_MIN_SNIPS", "24"))
    topic = args.topic
    candidates = retrieve(topic, topk=topk)
    filtered = []
    for c in candidates:
        if not c['text'].strip(): continue
        sents = select_sentences(c['text'], topic, max_sentences=8)
        if not sents: continue
        filtered.append({"doc_id": c['doc_id'], "page": c['page'], "text": " ".join(sents)})
    distinct_docs = {e["doc_id"] for e in filtered}
    total_snips = sum(ev["text"].count(". ") + 1 for ev in filtered)
    if len(distinct_docs) < MIN_DOCS or total_snips < MIN_SNIPS:
        print("[T2] refusal=insufficient_evidence_floor")
        write_run("T2", topic, candidates, {"refusal": True, "reason": "insufficient_evidence_floor"})
        return
    evidence_texts = []
    for e in filtered:
        header = f"[{e['doc_id']} p.{e['page']}]"
        evidence_texts.append(header + "\n- " + e["text"])
    reasoned = reason_over_evidence(evidence_texts, topic)
    try:
        json_part = reasoned[reasoned.find("{"):]
        obj = json.loads(json_part)
    except Exception:
        obj = None
    if obj and isinstance(obj, dict) and obj.get("positions"):
        to_validate = []
        for pos in obj["positions"]:
            for q in pos.get("representative_quotes", []):
                to_validate.append({"type": "quote", "doc_id": q["doc_id"], "page": q["page"], "text": q["quote"]})
        val = validate_evidence(to_validate, df)
        ok_map = {(v["item"]["doc_id"], v["item"]["page"], v["item"]["text"]): v["ok"] for v in val}
        kept_positions = []
        for pos in obj["positions"]:
            vq = []
            for q in pos.get("representative_quotes", []):
                if ok_map.get((q["doc_id"], q["page"], q["quote"]), False):
                    vq.append(q)
            pos["representative_quotes"] = vq
            if pos.get("supporting_docs") and vq:
                kept_positions.append(pos)
        obj["positions"] = kept_positions
        if not kept_positions:
            print("[T2] refusal=no_verifiable_evidence")
            write_run("T2", topic, filtered, {"refusal": True, "reason": "no_verifiable_evidence"})
            return
    run_id = uuid.uuid4().hex[:8]
    write_run("T2", topic, filtered, {"refusal": False, "reasoned": reasoned})
    print("\n[Reasoning output]\n", reasoned)
    print(f"[T2] run_id={run_id} refusal=False")

def t3(args, meta_path):
    df, refs = load_refs(meta_path)
    if not args.docpage:
        print("[T3] error: --docpage is required (format DOCID:PAGE)")
        return
    docpage = args.docpage
    candidates = retrieve(args.docpage.split(":")[0], topk=8)
    found = None
    for c in candidates:
        if f"{c['doc_id']}:{c['page']}".lower() == docpage.lower() or (c['doc_id']==docpage.split(':')[0] and str(c['page'])==docpage.split(':')[1]):
            text = c.get('text','').strip()
            if any(ch.isdigit() for ch in text):
                found = {"doc_id": c['doc_id'], "page": c['page'], "text": text}
                break
    if not found:
        print("[T3] refusal=table_not_found")
        write_run("T3", args.docpage, candidates, {"refusal": True})
        return
    run_id = uuid.uuid4().hex[:8]
    write_run("T3", args.docpage, [found], {"refusal": False})
    print(f"[T3] run_id={run_id} refusal=False")
    print("Extracted page text (first 800 chars):\n", found['text'][:800].replace("\n","\n"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=["t1","t2","t3"], help="which demo task to run")
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--claim")
    ap.add_argument("--topic")
    ap.add_argument("--docpage")
    ap.add_argument("--multi", action="store_true")
    ap.add_argument("--narrative-only", action="store_true",
                    help="Write/print the narrative review and skip appendix cards.")
    args = ap.parse_args()
    if args.task == "t1":
        t1(args, args.metadata)
    elif args.task == "t2":
        t2(args, args.metadata)
    elif args.task == "t3":
        t3(args, args.metadata)

if __name__ == "__main__":
    main()
