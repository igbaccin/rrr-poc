import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import rrr.render as render
from rrr.cli import ingest_main
from rrr.ingest import ExtractedMeta
from rrr.product_workspace import PreparedWorkspace, _warnings


class CheckCitationLabelsTests(unittest.TestCase):
    def tearDown(self):
        render.set_metadata_labels([])

    def test_a_sound_catalogue_has_no_problem(self):
        rows = [
            {"doc_id": "Plange_1979a", "year": "1979a"},
            {"doc_id": "Plange_1979b", "year": "1979b"},
            {"doc_id": "vanderGeest_2011", "year": 2011},
            {"doc_id": "Frankema&vanWaijenburg_2012", "year": 2012},
            {"doc_id": "AppiahKubiEtAl_2008", "year": 2008},
        ]
        self.assertEqual({"checked": 5, "problems": []}, render.check_citation_labels(rows))

    def test_the_catalogue_of_the_withheld_pilots_now_passes(self):
        rows = [
            {"doc_id": "Plange_1979", "year": 1979},
            {"doc_id": "Plange_1979a", "year": 1979},
            {"doc_id": "GhanaStatisticalService_2000", "year": 2000},
            {"doc_id": "GhanaStatisticalService_2000a", "year": 2000},
            {"doc_id": "vanderGeest_2011", "year": 2011},
        ]
        self.assertEqual([], render.check_citation_labels(rows)["problems"])

    def test_two_documents_under_one_label_are_reported_with_the_remedy(self):
        rows = [
            {"doc_id": "ReportNorth_2000", "display_label": "Service (2000)", "year": 2000},
            {"doc_id": "ReportSouth_2000", "display_label": "Service (2000)", "year": 2000},
            {"doc_id": "Brukum_1998", "year": 1998},
        ]
        result = render.check_citation_labels(rows)
        self.assertEqual(3, result["checked"])
        self.assertEqual(1, len(result["problems"]))
        problem = result["problems"][0]
        self.assertEqual("shared_label", problem["problem"])
        self.assertEqual("Service (2000)", problem["label"])
        self.assertEqual(["ReportNorth_2000", "ReportSouth_2000"], problem["doc_ids"])
        self.assertIn("year letter", problem["remedy"])

    def test_a_label_without_a_year_is_unreadable(self):
        rows = [
            {"doc_id": "Abstract_1961", "display_label": "Statistical abstract", "year": 1961},
            {"doc_id": "Brukum_1998", "year": 1998},
        ]
        problems = render.check_citation_labels(rows)["problems"]
        self.assertEqual(1, len(problems))
        self.assertEqual("unreadable_label", problems[0]["problem"])
        self.assertEqual(["Abstract_1961"], problems[0]["doc_ids"])
        self.assertIn("display_label", problems[0]["remedy"])

    def test_the_loaded_label_table_is_left_as_it_was(self):
        render.set_metadata_labels([{"doc_id": "North_1989", "year": 1989}])
        render.note_rendered_citation("North_1989", 4)
        table = dict(render._METADATA_LABEL_LOOKUP)
        rendered = dict(render._RENDERED_CITATIONS)
        render.check_citation_labels([{"doc_id": "Brukum_1998", "year": 1998}])
        self.assertEqual(table, render._METADATA_LABEL_LOOKUP)
        self.assertEqual(rendered, render._RENDERED_CITATIONS)
        self.assertEqual("North (1989)", render._doc_id_to_author_label("North_1989"))


class IngestLabelCheckTests(unittest.TestCase):
    def _run(self, rows, root):
        output = root / "metadata.csv"
        errors = io.StringIO()
        code = 0
        with patch("rrr.ingest.ingest_corpus", return_value=rows):
            with contextlib.redirect_stderr(errors), contextlib.redirect_stdout(io.StringIO()):
                try:
                    ingest_main(["--corpus", str(root), "--output", str(output)])
                except SystemExit as stop:
                    code = stop.code
        report = json.loads((root / "ingest_report.json").read_text(encoding="utf-8"))
        return code, report, errors.getvalue()

    def _row(self, doc_id, year, label=""):
        return ExtractedMeta(
            doc_id=doc_id, year=year, pdf_path=f"corpus/{doc_id}.pdf",
            confidence="high", source="bib_sidecar", display_label=label,
        )

    def test_a_sound_catalogue_is_admitted_and_the_check_is_recorded(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            code, report, errors = self._run(
                [self._row("Plange_1979a", "1979a"), self._row("Plange_1979b", "1979b")],
                Path(temp_dir),
            )
        self.assertEqual(0, code)
        self.assertEqual({"checked": 2, "problems": []}, report["label_check"])
        self.assertEqual("", errors)

    def test_an_unreadable_label_stops_before_indexing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            code, report, errors = self._run(
                [
                    self._row("Abstract_1961", "1961", label="Statistical abstract"),
                    self._row("Brukum_1998", "1998"),
                ],
                Path(temp_dir),
            )
        self.assertEqual(3, code)
        self.assertEqual("unreadable_label", report["label_check"]["problems"][0]["problem"])
        status = {row["doc_id"]: row["status"] for row in report["rows"]}
        self.assertEqual("correction_required", status["Abstract_1961"])
        self.assertEqual("approved", status["Brukum_1998"])
        notes = {row["doc_id"]: row["notes"] for row in report["rows"]}
        self.assertIn("citation label cannot be read back", notes["Abstract_1961"])
        self.assertIn("citation label cannot be read back: Statistical abstract", errors)

    def test_a_shared_label_is_a_warning_and_does_not_stop_ingest(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            code, report, errors = self._run(
                [
                    self._row("ReportNorth_2000", "2000", label="Service (2000)"),
                    self._row("ReportSouth_2000", "2000", label="Service (2000)"),
                ],
                root,
            )
            warnings = _warnings(root)
        self.assertEqual(0, code)
        self.assertIn("warning: 2 documents share the citation label Service (2000)", errors)
        self.assertEqual(1, len(warnings))
        self.assertEqual("shared_citation_label", warnings[0]["type"])
        self.assertEqual(["ReportNorth_2000", "ReportSouth_2000"], warnings[0]["doc_ids"])
        self.assertIn("year letter", warnings[0]["message"])

    def test_prepare_result_carries_the_warnings(self):
        warning = {"type": "shared_citation_label", "message": "two documents share a label"}
        prepared = PreparedWorkspace(
            corpus_dir=Path("corpus"), workspace_dir=Path("work"),
            metadata_path=Path("work") / "metadata.csv", reused=False, pdf_count=2,
            warnings=(warning,),
        )
        self.assertEqual([warning], prepared.as_dict()["warnings"])


if __name__ == "__main__":
    unittest.main()
