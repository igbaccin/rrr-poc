import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from _pdf_fixtures import make_layout_pdf

from rrr.preprocess import (
    _bibliography_run_start,
    resolve_reference_leaks,
    scan_reference_leaks,
)
from rrr.product_workspace import _warnings


PROSE = (
    "Seasonal migration from the northern savanna to the cocoa belt and the "
    "mines rose with every tax demand and every failed harvest. District "
    "officers recorded the departures, the chiefs who were told to supply "
    "carriers, and the wages that came back as cloth and bicycles. The record "
    "is thin for the years of the war and fuller after the labour ordinances. "
)
ENTRIES = "\n".join([
    "Bennett, D. (1980), The Future of Welfare Geography in Canada and Beyond.",
    "Darkoh, M. B. (1976), Recent Trends in Regional Planning in Ghana, Accra.",
    "Dickson, K. B. (1970), The Population Factor in the Development of Ghana.",
    "Ewusi, K. (1976), Disparities in Levels of Regional Development in Ghana.",
    "Forde, E. R. (1968), The Population of Ghana: A Study of Spatial Relations.",
    "Gaisie, S. K. (1974), Population Growth and its Components in Ghana, Legon.",
    "Harvey, D. (1973), Social Justice and the City. New York: Edward Arnold.",
    "Larson, D. (1979), The PQLI: Measuring the Condition of the World's Poor.",
    "Morris, M. D. (1979), Measuring the Condition of the World's Poor, Oxford.",
    "Rawls, J. (1971), A Theory of Justice. Cambridge: Harvard University Press.",
    "Smith, D. M. (1979), Where the Grass is Greener: An Unequal World, London.",
    "Streeten, P. P. (1981), First Things First: Meeting Basic Human Needs.",
])


def _write_page(root: Path, name: str, text: str) -> None:
    for folder in ("page_text", "page_meta"):
        (root / "data" / folder).mkdir(parents=True, exist_ok=True)
    (root / "data" / "page_text" / f"{name}.txt").write_text(text, encoding="utf-8")
    (root / "data" / "page_meta" / f"{name}.json").write_text(
        json.dumps({"doc_id": name.rsplit("_page_", 1)[0], "char_count": len(text)}),
        encoding="utf-8",
    )


class ResolveReferenceLeakTests(unittest.TestCase):
    def test_a_page_with_text_and_a_list_is_cut_at_the_list(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write_page(root, "Thomas_1973_page_4", PROSE + "\n" + ENTRIES)
            _write_page(root, "Thomas_1973_page_5", PROSE)
            page_text = root / "data" / "page_text"
            self.assertEqual(1, len(scan_reference_leaks(page_text)))
            records = resolve_reference_leaks(page_text)
            kept = (page_text / "Thomas_1973_page_4.txt").read_text(encoding="utf-8")
            meta = json.loads(
                (root / "data" / "page_meta" / "Thomas_1973_page_4.json").read_text(encoding="utf-8")
            )
            untouched = (page_text / "Thomas_1973_page_5.txt").read_text(encoding="utf-8")
            remaining = scan_reference_leaks(page_text)
        self.assertEqual(PROSE.rstrip(), kept)
        self.assertEqual(PROSE, untouched)
        self.assertEqual([], remaining)
        self.assertEqual(1, len(records))
        record = records[0]
        self.assertEqual(("Thomas_1973", 4, "cut"), (record["doc_id"], record["page"], record["action"]))
        self.assertEqual(len(PROSE.rstrip()), record["kept_chars"])
        self.assertEqual(5, len(record["lines"]))
        self.assertTrue(record["lines"][0].startswith("Bennett, D. (1980)"))
        self.assertEqual(len(kept), meta["char_count"])
        self.assertEqual(4, meta["page"])

    def test_a_page_that_is_all_list_is_set_aside_and_keeps_its_number(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write_page(root, "Thomas_1973_page_3", PROSE)
            _write_page(root, "Thomas_1973_page_4", ENTRIES)
            _write_page(root, "Thomas_1973_page_5", PROSE)
            page_text = root / "data" / "page_text"
            records = resolve_reference_leaks(page_text)
            left = sorted(path.name for path in page_text.glob("*.txt"))
            aside = root / "data" / "page_text_set_aside" / "Thomas_1973_page_4.txt"
            aside_text = aside.read_text(encoding="utf-8")
            meta_left = sorted(path.name for path in (root / "data" / "page_meta").glob("*.json"))
        self.assertEqual(["Thomas_1973_page_3.txt", "Thomas_1973_page_5.txt"], left)
        self.assertEqual(["Thomas_1973_page_3.json", "Thomas_1973_page_5.json"], meta_left)
        self.assertEqual(ENTRIES, aside_text)
        self.assertEqual("set_aside", records[0]["action"])
        self.assertEqual(0, records[0]["kept_chars"])

    def test_a_heading_before_the_list_is_where_the_page_is_cut(self):
        text = PROSE + "\nBibliography:\n" + "\n".join(ENTRIES.splitlines()[:4])
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write_page(root, "Doc_2001_page_9", text)
            records = resolve_reference_leaks(root / "data" / "page_text")
            kept = (root / "data" / "page_text" / "Doc_2001_page_9.txt").read_text(encoding="utf-8")
        self.assertEqual("cut", records[0]["action"])
        self.assertEqual(PROSE.rstrip(), kept)

    def test_one_line_of_that_shape_in_running_text_starts_no_run(self):
        text = (
            "Accra, Ghana, where the survey began in the dry season of that year.\n"
            + PROSE + "\n" + ENTRIES
        )
        self.assertEqual(text.index("Bennett, D."), _bibliography_run_start(text))

    def test_a_clean_corpus_yields_no_record(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write_page(root, "Thomas_1973_page_1", PROSE)
            self.assertEqual([], resolve_reference_leaks(root / "data" / "page_text"))


class PreprocessCommandTests(unittest.TestCase):
    def _workspace(self, root: Path) -> Path:
        prose_page = [((40, 30, 430, 640), PROSE * 4)]
        list_page = [
            ((40, 30, 430, 200), PROSE * 2),
            ((40, 220, 430, 640), ENTRIES),
        ]
        pdf = make_layout_pdf(
            root / "papers" / "Thomas (1973).pdf",
            [prose_page, prose_page, list_page, prose_page, prose_page, prose_page],
        )
        metadata = root / "work" / "metadata.csv"
        metadata.parent.mkdir(parents=True)
        metadata.write_text(f"doc_id,pdf_path\nThomas_1973,{pdf}\n", encoding="utf-8")
        (root / "work" / "ingest_report.json").write_text(
            json.dumps({"n_pdfs": 1, "rows": []}), encoding="utf-8",
        )
        return metadata

    def _run(self, metadata: Path, *extra, strict_env=None):
        env = os.environ.copy()
        env["RRR_PROJECT_ROOT"] = str(metadata.parent)
        env["PYTHONIOENCODING"] = "utf-8"
        env.pop("RRR_STRICT_PREPARE", None)
        if strict_env is not None:
            env["RRR_STRICT_PREPARE"] = strict_env
        options = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
        return subprocess.run(
            [sys.executable, "-m", "rrr.preprocess", "--metadata", str(metadata),
             "--workers", "1", *extra],
            env=env, capture_output=True, text=True, encoding="utf-8",
            errors="replace", **options,
        )

    def test_default_cuts_the_page_reports_it_and_finishes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            metadata = self._workspace(root)
            completed = self._run(metadata)
            report = json.loads(
                (root / "work" / "ingest_report.json").read_text(encoding="utf-8")
            )
            page_text = root / "work" / "data" / "page_text"
            pages = sorted(path.name for path in page_text.glob("*.txt"))
            page3 = (page_text / "Thomas_1973_page_3.txt").read_text(encoding="utf-8")
            warnings = _warnings(root / "work")
        self.assertEqual(0, completed.returncode, completed.stdout + completed.stderr)
        self.assertIn("[invariant] OK", completed.stdout)
        self.assertIn("warning: Thomas_1973 page 3: a reference list remained", completed.stderr)
        self.assertIn("Bennett, D. (1980)", completed.stderr)
        self.assertEqual(6, len(pages))
        self.assertNotIn("Bennett", page3)
        self.assertIn("Seasonal migration", page3)
        self.assertEqual(1, report["n_pdfs"])
        self.assertEqual(1, len(report["reference_leaks"]))
        leak = report["reference_leaks"][0]
        self.assertEqual(("Thomas_1973", 3, "cut"), (leak["doc_id"], leak["page"], leak["action"]))
        self.assertEqual(1, len(warnings))
        self.assertEqual("reference_list_removed", warnings[0]["type"])
        self.assertEqual(("Thomas_1973", 3), (warnings[0]["doc_id"], warnings[0]["page"]))
        self.assertIn("Thomas_1973 page 3 still held a reference list", warnings[0]["message"])

    def test_strict_keeps_the_hard_stop(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            metadata = self._workspace(Path(temp_dir))
            flagged = self._run(metadata, "--strict")
            by_env = self._run(metadata, strict_env="1")
        for completed in (flagged, by_env):
            self.assertEqual(1, completed.returncode)
            self.assertIn("[FATAL] reference-leak invariant FAILED - 1 retained", completed.stdout)
            self.assertIn("Thomas_1973_page_3.txt", completed.stdout)
            completed.stdout.encode("ascii")


if __name__ == "__main__":
    unittest.main()
