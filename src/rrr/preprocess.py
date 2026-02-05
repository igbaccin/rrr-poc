import argparse, os, pandas as pd, re
from pdfminer.high_level import extract_text
from rrr.utils import sha256_file, ensure_dir, save_json
from multiprocessing import Pool, cpu_count
def extract_pages(pdf_path: str):
    text = extract_text(pdf_path)
    pages = text.split("\x0c")
    pages = [p.strip() for p in pages if p.strip()]
    # clean CID / control characters
    pages = [re.sub(r'\(cid:\d+\)', '', p) for p in pages]
    pages = [re.sub(r'[\x00-\x1f\x7f-\x9f]', '', p) for p in pages]
    return pages
def _process_one(row_dict):
    pdf = row_dict.get("pdf_path")
    doc_id = str(row_dict.get("doc_id"))
    if not (isinstance(pdf, str) and os.path.isfile(pdf)):
        return {"doc_id": doc_id, "ok": False, "reason": "missing_pdf"}
    try:
        h = sha256_file(pdf)
        pages = extract_pages(pdf)
        for i, ptxt in enumerate(pages, start=1):
            outp = os.path.join("data/page_text", f"{doc_id}_page_{i}.txt")
            ensure_dir(os.path.dirname(outp))
            with open(outp, "w", encoding="utf-8") as f:
                f.write(ptxt)
        meta = {"doc_id": doc_id, "pdf_path": pdf, "hash": h, "page_count": len(pages)}
        save_json(meta, f"data/{doc_id}.json")
        return {"doc_id": doc_id, "ok": True, "pages": len(pages), "hash": h[:12]}
    except Exception as e:
        return {"doc_id": doc_id, "ok": False, "reason": str(e)}
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--workers", type=int, default=max(1, cpu_count()-1))
    args = ap.parse_args()
    df = pd.read_csv(args.metadata)
    rows = [r.to_dict() for _, r in df.iterrows()]
    with Pool(processes=args.workers) as pool:
        for res in pool.imap_unordered(_process_one, rows, chunksize=1):
            if res.get("ok"):
                print(f"[ok] {res['doc_id']}: {res['pages']} pages, hash={res['hash']}...")
            else:
                print(f"[skip] {res['doc_id']}: {res.get('reason')}")
    print("[done] preprocessing")
if __name__ == "__main__":
    main()
