import contextlib
import io
import json
import random
import tempfile
import unittest
from pathlib import Path

from _pdf_fixtures import make_pdf, write_bib

from rrr.cli import ingest_main
from rrr.ingest import (
    ExtractedMeta,
    assign_year_letters,
    ingest_corpus,
    year_letter_groups,
)
from rrr.product_workspace import PreparedWorkspace, _same_author_year


def _row(doc_id, year, pdf, bib_key="", sha="0"):
    return ExtractedMeta(
        doc_id=doc_id, year=year, pdf_path=str(Path("corpus") / pdf),
        bib_key=bib_key, content_sha1=sha, source="bib_sidecar", confidence="high",
    )


def _ids(rows):
    return {Path(row.pdf_path).name: (row.doc_id, row.year) for row in rows}


class AssignYearLettersTests(unittest.TestCase):
    def test_the_pilot_pair_is_lettered_from_its_file_names(self):
        rows = [
            _row("Plange_1979", "1979", "Plange_1979a.pdf"),
            _row("Plange_1979a", "1979", "Plange_1979b.pdf"),
        ]
        groups = assign_year_letters(rows)
        self.assertEqual(
            {
                "Plange_1979a.pdf": ("Plange_1979a", "1979a"),
                "Plange_1979b.pdf": ("Plange_1979b", "1979b"),
            },
            _ids(rows),
        )
        self.assertEqual("Plange (1979)", groups[0]["label"])
        self.assertEqual(
            ["file name", "file name"],
            [member["letter_source"] for member in groups[0]["members"]],
        )

    def test_surname_year_file_names_carry_the_letter(self):
        rows = [
            _row("GhanaStatisticalService_2000", "2000", "Ghana Statistical Service (2000b).pdf"),
            _row("GhanaStatisticalService_2000a", "2000", "Ghana Statistical Service (2000a).pdf"),
        ]
        assign_year_letters(rows)
        self.assertEqual(
            {
                "Ghana Statistical Service (2000a).pdf": ("GhanaStatisticalService_2000a", "2000a"),
                "Ghana Statistical Service (2000b).pdf": ("GhanaStatisticalService_2000b", "2000b"),
            },
            _ids(rows),
        )

    def test_letters_without_any_hint_follow_file_name_order_on_every_run(self):
        expected = {
            "alpha.pdf": ("Plange_1979a", "1979a"),
            "beta.pdf": ("Plange_1979b", "1979b"),
            "gamma.pdf": ("Plange_1979c", "1979c"),
        }
        for seed in range(6):
            rows = [
                _row("Plange_1979", "1979", "beta.pdf"),
                _row("Plange_1979a", "1979", "gamma.pdf"),
                _row("Plange_1979b", "1979", "alpha.pdf"),
            ]
            random.Random(seed).shuffle(rows)
            assign_year_letters(rows)
            self.assertEqual(expected, _ids(rows))
            self.assertTrue(all(row.year_letter_source == "assigned" for row in rows))

    def test_a_bibliography_year_letter_is_kept_and_the_other_work_gets_one_too(self):
        rows = [
            _row("Plange_1979", "1979", "first.pdf"),
            _row("Plange_1979b", "1979b", "second.pdf"),
        ]
        assign_year_letters(rows)
        self.assertEqual(
            {
                "first.pdf": ("Plange_1979a", "1979a"),
                "second.pdf": ("Plange_1979b", "1979b"),
            },
            _ids(rows),
        )
        sources = {Path(row.pdf_path).name: row.year_letter_source for row in rows}
        self.assertEqual("assigned", sources["first.pdf"])
        self.assertEqual("bibliography year", sources["second.pdf"])

    def test_a_bibliography_key_letter_is_kept(self):
        rows = [
            _row("Plange_1979", "1979", "one.pdf", bib_key="Plange1979b"),
            _row("Plange_1979a", "1979", "two.pdf", bib_key="Plange_1979a"),
        ]
        assign_year_letters(rows)
        self.assertEqual(
            {"one.pdf": ("Plange_1979b", "1979b"), "two.pdf": ("Plange_1979a", "1979a")},
            _ids(rows),
        )

    def test_two_works_that_claim_one_letter_still_end_up_apart(self):
        rows = [
            _row("Plange_1979a", "1979a", "second.pdf"),
            _row("Plange_1979a", "1979a", "first.pdf"),
        ]
        assign_year_letters(rows)
        ids = _ids(rows)
        self.assertEqual(("Plange_1979a", "1979a"), ids["first.pdf"])
        self.assertEqual(("Plange_1979b", "1979b"), ids["second.pdf"])
        self.assertIn("already in use", rows[0].year_letter_source)

    def test_a_single_work_is_left_as_it_is(self):
        rows = [
            _row("Plange_1979a", "1979a", "Plange (1979a).pdf"),
            _row("Brukum_1998", "1998", "Brukum (1998).pdf"),
        ]
        self.assertEqual([], assign_year_letters(rows))
        self.assertEqual(
            {
                "Plange (1979a).pdf": ("Plange_1979a", "1979a"),
                "Brukum (1998).pdf": ("Brukum_1998", "1998"),
            },
            _ids(rows),
        )

    def test_different_co_authors_need_no_letter(self):
        rows = [
            _row("Aryeetey&Kanbur_2008", "2008", "a.pdf"),
            _row("Aryeetey&McKay_2008", "2008", "b.pdf"),
        ]
        self.assertEqual([], assign_year_letters(rows))
        self.assertEqual({"2008"}, {row.year for row in rows})

    def test_rows_without_a_document_id_are_ignored(self):
        rows = [
            ExtractedMeta(pdf_path="corpus/scan.pdf", source="needs_ocr"),
            _row("Plange_1979", "1979", "a.pdf"),
        ]
        self.assertEqual([], assign_year_letters(rows))

    def test_groups_can_be_rebuilt_from_lettered_rows(self):
        rows = [
            _row("Plange_1979", "1979", "Plange (1979a).pdf"),
            _row("Plange_1979a", "1979", "Plange (1979b).pdf"),
            _row("Brukum_1998", "1998", "Brukum (1998).pdf"),
        ]
        first = assign_year_letters(rows)
        self.assertEqual(first, year_letter_groups(rows))


class IngestYearLetterEndToEndTests(unittest.TestCase):
    TITLE_A = "Underdevelopment in Northern Ghana: Natural Causes or Colonial Capitalism"
    TITLE_B = "Opportunity Cost and Labour Migration: A Misinterpretation of Proletarianisation"

    def _corpus(self, root, bib_years=("1979", "1979")):
        corpus = root / "papers"
        make_pdf(corpus / "Plange (1979a).pdf", self.TITLE_A, "Nii-K. Plange", "1979")
        make_pdf(corpus / "Plange (1979b).pdf", self.TITLE_B, "Nii-K. Plange", "1979")
        make_pdf(
            corpus / "Brukum (1998).pdf",
            "Studied Neglect or Lack of Resources in the Northern Territories",
            "N. J. K. Brukum", "1998",
        )
        write_bib(corpus / "bibliography.bib", [
            ("article", "PlangeOne", {
                "author": "Plange, Nii-K.", "title": self.TITLE_A,
                "year": bib_years[0], "journal": "Review of African Political Economy",
            }),
            ("article", "PlangeTwo", {
                "author": "Plange, Nii-K.", "title": self.TITLE_B,
                "year": bib_years[1], "journal": "The Journal of Modern African Studies",
            }),
            ("article", "Brukum1998", {
                "author": "Brukum, N. J. K.",
                "title": "Studied Neglect or Lack of Resources in the Northern Territories",
                "year": "1998", "journal": "Transactions of the Historical Society of Ghana",
            }),
        ])
        return corpus

    def test_ingest_never_emits_the_shifted_pair(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            corpus = self._corpus(Path(temp_dir))
            rows = ingest_corpus(
                corpus, sidecar_bib=corpus / "bibliography.bib",
                use_llm=False, use_crossref=False, use_openalex=False, progress=False,
            )
        ids = _ids(rows)
        self.assertEqual(("Plange_1979a", "1979a"), ids["Plange (1979a).pdf"])
        self.assertEqual(("Plange_1979b", "1979b"), ids["Plange (1979b).pdf"])
        self.assertEqual(("Brukum_1998", "1998"), ids["Brukum (1998).pdf"])
        self.assertNotIn("Plange_1979", {row.doc_id for row in rows})

    def test_ingest_report_and_console_list_the_pair(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            corpus = self._corpus(root)
            output = root / "work" / "metadata.csv"
            console = io.StringIO()
            with contextlib.redirect_stdout(console):
                ingest_main([
                    "--corpus", str(corpus), "--output", str(output),
                    "--bib", str(corpus / "bibliography.bib"),
                    "--no-llm", "--no-crossref", "--no-openalex",
                ])
            report = json.loads((root / "work" / "ingest_report.json").read_text(encoding="utf-8"))
            catalogue = output.read_text(encoding="utf-8")
            pairs = _same_author_year(root / "work")
        self.assertEqual(1, len(report["same_author_year"]))
        group = report["same_author_year"][0]
        self.assertEqual("Plange (1979)", group["label"])
        self.assertEqual(
            ["Plange_1979a", "Plange_1979b"],
            [member["doc_id"] for member in group["members"]],
        )
        self.assertEqual(["1979a", "1979b"], [member["year"] for member in group["members"]])
        self.assertIn("same author and year, Plange (1979)", console.getvalue())
        self.assertIn("Plange_1979a,", catalogue)
        self.assertIn(",1979a,", catalogue)
        self.assertEqual((group,), pairs)

    def test_prepare_result_carries_the_pairs(self):
        group = {"label": "Plange (1979)", "members": [{"doc_id": "Plange_1979a"}]}
        prepared = PreparedWorkspace(
            corpus_dir=Path("corpus"), workspace_dir=Path("work"),
            metadata_path=Path("work") / "metadata.csv", reused=False, pdf_count=3,
            same_author_year=(group,),
        )
        self.assertEqual([group], prepared.as_dict()["same_author_year"])


if __name__ == "__main__":
    unittest.main()
