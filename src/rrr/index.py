import argparse, os, glob, pickle, numpy as np, pandas as pd, re
from rank_bm25 import BM25Okapi
from rrr.utils import ensure_dir

_tok = re.compile(r"[A-Za-z0-9]+")
def _tokenize(txt: str):
    return _tok.findall((txt or "").lower())

def _iter_pages_for(doc_id):
    for tpath in sorted(glob.glob(f"data/page_text/{doc_id}_page_*.txt")):
        yield tpath

def build_bm25(doc_id_list):
    docs, page_ids = [], []
    for doc_id in doc_id_list:
        for tpath in _iter_pages_for(doc_id):
            with open(tpath, "r", encoding="utf-8") as f:
                txt = f.read()
            toks = _tokenize(txt)
            docs.append(toks); page_ids.append(os.path.basename(tpath).replace(".txt",""))
    bm = BM25Okapi(docs)
    return bm, page_ids

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", required=True)
    args = ap.parse_args()
    df = pd.read_csv(args.metadata)
    doc_ids = [str(x) for x in df["doc_id"].dropna().unique().tolist()]
    bm, page_ids = build_bm25(doc_ids)
    ensure_dir("indices")
    with open("indices/bm25.pkl","wb") as f:
        pickle.dump(bm, f, protocol=pickle.HIGHEST_PROTOCOL)
    np.save("indices/page_ids.npy", np.array(page_ids, dtype=object))
    df[["doc_id","pdf_path"]].to_csv("indices/docs.csv", index=False)
    print(f"[ok] bm25 persisted: {len(page_ids)} pages")
if __name__ == "__main__":
    main()
