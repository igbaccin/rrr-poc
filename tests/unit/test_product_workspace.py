import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rrr.cli import ingest_main
from rrr.ingest import ExtractedMeta
from rrr.product_workspace import (
    WORKSPACE_SCHEMA,
    CorpusDiscoveryError,
    _is_reusable,
    bibliography_snapshot,
    corpus_snapshot,
    discover_corpus_dir,
    workspace_for_corpus,
)


class ProductWorkspaceTests(unittest.TestCase):
    def test_discovers_the_only_nearby_pdf_collection(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            papers = root / "my papers"
            papers.mkdir()
            (papers / "one.PDF").write_bytes(b"pdf")
            self.assertEqual(discover_corpus_dir(root), papers.resolve())

    def test_reports_ambiguous_pdf_collections(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for name in ("set-a", "set-b"):
                folder = root / name
                folder.mkdir()
                (folder / "paper.pdf").write_bytes(b"pdf")
            with self.assertRaises(CorpusDiscoveryError) as raised:
                discover_corpus_dir(root)
            self.assertEqual(len(raised.exception.candidates), 2)

    def test_conventional_corpus_folder_takes_priority(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            corpus = root / "corpus"
            corpus.mkdir()
            (corpus / "paper.pdf").write_bytes(b"pdf")
            another = root / "appendix"
            another.mkdir()
            (another / "supplement.pdf").write_bytes(b"pdf")
            self.assertEqual(discover_corpus_dir(root), corpus.resolve())

    def test_missing_collection_uses_user_facing_guidance(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(CorpusDiscoveryError, "Select or open"):
                discover_corpus_dir(Path(temp_dir))

    def test_prepare_json_keeps_machine_output_parseable(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            completed = subprocess.run(
                [sys.executable, "-m", "rrr.cli", "prepare", "--json"],
                cwd=temp_dir,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 2)
            payload = json.loads(completed.stdout)
            self.assertIn("No PDF collection was found", payload["error"])

    def test_managed_workspace_is_stable_and_outside_the_corpus(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            corpus = root / "papers"
            corpus.mkdir()
            managed = root / "managed"
            first = workspace_for_corpus(corpus, managed)
            second = workspace_for_corpus(corpus, managed)
            self.assertEqual(first, second)
            self.assertEqual(first.parent, managed / "corpora")
            self.assertFalse(first.is_relative_to(corpus))

    def test_reuse_requires_matching_snapshot_and_complete_assets(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            corpus = root / "papers"
            workspace = root / "workspace"
            corpus.mkdir()
            workspace.mkdir()
            (corpus / "paper.pdf").write_bytes(b"pdf")
            snapshot = corpus_snapshot(corpus)
            (workspace / "metadata.csv").write_text("doc_id,pdf_path\n", encoding="utf-8")
            (workspace / "indices").mkdir()
            for name in ("bm25.pkl", "page_ids.npy", "docs.csv"):
                (workspace / "indices" / name).write_bytes(b"x")
            (workspace / "data" / "page_text").mkdir(parents=True)
            (workspace / "data" / "page_text" / "paper_page_1.txt").write_text(
                "text", encoding="utf-8"
            )
            (workspace / "workspace.json").write_text(
                json.dumps({
                    "schema_version": WORKSPACE_SCHEMA,
                    "corpus_snapshot": snapshot,
                    "bibliography": None,
                }),
                encoding="utf-8",
            )
            self.assertTrue(_is_reusable(workspace, snapshot))
            (corpus / "paper.pdf").write_bytes(b"changed")
            self.assertFalse(_is_reusable(workspace, corpus_snapshot(corpus)))

    def _prepared(self, root, bibliography, schema=WORKSPACE_SCHEMA):
        corpus = root / "papers"
        workspace = root / "workspace"
        corpus.mkdir(exist_ok=True)
        workspace.mkdir(exist_ok=True)
        (corpus / "paper.pdf").write_bytes(b"pdf")
        (workspace / "metadata.csv").write_text("doc_id,pdf_path\n", encoding="utf-8")
        (workspace / "indices").mkdir(exist_ok=True)
        for name in ("bm25.pkl", "page_ids.npy", "docs.csv"):
            (workspace / "indices" / name).write_bytes(b"x")
        (workspace / "data" / "page_text").mkdir(parents=True, exist_ok=True)
        (workspace / "data" / "page_text" / "paper_page_1.txt").write_text("text", encoding="utf-8")
        snapshot = corpus_snapshot(corpus)
        manifest = {"corpus_snapshot": snapshot, "bibliography": bibliography_snapshot(bibliography)}
        if schema is not None:
            manifest["schema_version"] = schema
        (workspace / "workspace.json").write_text(json.dumps(manifest), encoding="utf-8")
        return workspace, snapshot

    def test_an_edited_bibliography_ends_the_reuse(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            bib = root / "bibliography.bib"
            bib.write_text("@article{a, year = {1979}}\n", encoding="utf-8")
            workspace, snapshot = self._prepared(root, bib)
            self.assertTrue(_is_reusable(workspace, snapshot, bibliography_snapshot(bib)))
            bib.write_text("@article{a, year = {1979a}}\n", encoding="utf-8")
            self.assertFalse(_is_reusable(workspace, snapshot, bibliography_snapshot(bib)))
            other = root / "other.bib"
            other.write_text("@article{a, year = {1979}}\n", encoding="utf-8")
            self.assertFalse(_is_reusable(workspace, snapshot, bibliography_snapshot(other)))
            self.assertFalse(_is_reusable(workspace, snapshot, None))

    def test_a_workspace_from_an_older_rrr_is_prepared_again(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for schema in (None, 1):
                workspace, snapshot = self._prepared(root, None, schema=schema)
                self.assertFalse(_is_reusable(workspace, snapshot, None), schema)
            workspace, snapshot = self._prepared(root, None)
            self.assertTrue(_is_reusable(workspace, snapshot, None))

    def test_a_missing_bibliography_has_no_snapshot(self):
        self.assertIsNone(bibliography_snapshot(None))
        self.assertIsNone(bibliography_snapshot(Path("no") / "such.bib"))

    def test_approval_cannot_admit_a_row_without_a_document_id(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            corpus = root / "papers"
            corpus.mkdir()
            output = root / "metadata.csv"
            unresolved = ExtractedMeta(
                pdf_path=str(corpus / "scan.pdf"),
                confidence="low",
                source="needs_ocr",
            )
            with patch("rrr.ingest.ingest_corpus", return_value=[unresolved]):
                with self.assertRaises(SystemExit) as raised:
                    ingest_main(
                        [
                            "--corpus",
                            str(corpus),
                            "--output",
                            str(output),
                            "--accept-low-confidence",
                        ]
                    )
            self.assertEqual(raised.exception.code, 3)
            report = json.loads((root / "ingest_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["rows"][0]["status"], "correction_required")


if __name__ == "__main__":
    unittest.main()
