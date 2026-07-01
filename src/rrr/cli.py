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
    # claim, no writer) and t3 (page extraction) entrypoints — t3 stays retired
    # (v13 removed). t1 REVIVED in v15.9 as a claim-evaluator mode: same
    # pipeline as T2 up to Stage 2 posture, but stops before the writer and
    # emits claim_verdict.md instead of review_composed.md.
    if not getattr(args, "multi", False):
        raise SystemExit(
            "cli.t2 now requires --multi. The single-pass path was retired in v13; "
            "use scripts/run_small_validation.py or scripts/run_battery.sh for the "
            "full layered pipeline."
        )
    layered_t2(args, meta_path)


def t1(args, meta_path):
    """v15.9 revival: claim-eval mode. Same pipeline as T2 up to Stage 2
    posture, then stops. Emits runs/claim_verdict.md with per-cluster + aggregate
    breakdown of how the corpus stands on the claim. ~15-20% of T2's runtime
    because it skips the writer + validation chain."""
    # We reuse layered_t2's whole entry, gated by args.t1_only which the
    # reasoner short-circuits on after Stage 2 + ledger write.
    setattr(args, "multi", True)
    setattr(args, "t1_only", True)
    # narrative-only saves us the T2_review.md appendix step we don't need.
    setattr(args, "narrative_only", True)
    layered_t2(args, meta_path)


def main():
    ap = argparse.ArgumentParser(
        description="RRR CLI. `t2` runs the full literature-review pipeline; "
                    "`t1` runs the claim-evaluator (same up to Stage 2, no writer)."
    )
    ap.add_argument("task", choices=["t1", "t2"],
                    help="Which pipeline task to run.")
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--topic", required=True,
                    help="Topic (T2) or claim (T1) to evaluate.")
    ap.add_argument("--multi", action="store_true",
                    help="Run the layered T2 pipeline (default for both t1 and t2).")
    ap.add_argument("--narrative-only", action="store_true",
                    help="T2: skip the T2_review.md appendix. Ignored by t1.")
    ap.add_argument("--linkify", action="store_true",
                    help="T2: rewrite in-text citations as clickable markdown links "
                         "to the source PDF page. Sets RRR_LINKIFY=1.")
    args = ap.parse_args()

    if args.linkify:
        import os
        os.environ["RRR_LINKIFY"] = "1"

    if args.task == "t1":
        t1(args, args.metadata)
    else:
        t2(args, args.metadata)


if __name__ == "__main__":
    main()
