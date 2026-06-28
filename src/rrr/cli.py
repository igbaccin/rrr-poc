import argparse, pandas as pd
from rrr.reasoner import layered_t2


def load_refs(meta_path):
    df = pd.read_csv(meta_path)
    refs = {}
    for _, r in df.iterrows():
        def safe(x): return str(x).strip() if isinstance(x, str) else ""
        y = f" ({safe(r.get('year'))})" if 'year' in r and pd.notna(r.get('year')) and safe(r.get('year')) else ""
        parts = []
        if safe(r.get('authors')): parts.append(safe(r.get('authors')) + y)
        if safe(r.get('title')):   parts.append(f"*{safe(r.get('title'))}*")
        if safe(r.get('venue')):   parts.append(safe(r.get('venue')))
        cite = ". ".join(parts).strip() or str(r.get('doc_id'))
        refs[str(r["doc_id"])] = cite
    return df, refs


def t2(args, meta_path):
    # v13: cli.t2 now only dispatches to the live multi-pass path. The legacy
    # single-pass T2 (`t2` without --multi) was the v6-era strict reasoner
    # entrypoint; the layered_t2 architecture (v8+) supersedes it end-to-end
    # and is what every battery script invokes. Same for the v6-era t1 (single
    # claim, no writer) and t3 (page extraction) entrypoints — both removed in
    # v13 because no caller in scripts/ or revision_notes/ targets them.
    if not getattr(args, "multi", False):
        raise SystemExit(
            "cli.t2 now requires --multi. The single-pass path was retired in v13; "
            "use scripts/run_small_validation.py or scripts/run_battery.sh for the "
            "full layered pipeline."
        )
    layered_t2(args, meta_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=["t2"],
                    help="Which pipeline task to run (only t2 --multi is supported in v13).")
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--topic")
    ap.add_argument("--multi", action="store_true",
                    help="Run the layered T2 pipeline (the only path retained in v13).")
    ap.add_argument("--narrative-only", action="store_true",
                    help="Write/print the narrative review and skip appendix cards.")
    args = ap.parse_args()
    if args.task == "t2":
        t2(args, args.metadata)


if __name__ == "__main__":
    main()
