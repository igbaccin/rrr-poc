import csv
import importlib.util
import json
import re
import tempfile
import unittest
from pathlib import Path

from _pdf_fixtures import make_pdf, write_bib

import rrr.render as render
import rrr.writer as writer
from rrr.ingest import build_display_label, ingest_corpus


REPO_ROOT = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location(
    "check_citations_for_labels", REPO_ROOT / "scripts" / "check_citations.py"
)
CHECKER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(CHECKER)


WORKS = [
    ("WorldBank_1995", ["World Bank"], "1995"),
    ("GhanaStatisticalService_2000a", ["Ghana Statistical Service"], "2000a"),
    ("GhanaStatisticalService_2000b", ["Ghana Statistical Service"], "2000b"),
    ("Plange_1979a", ["Plange"], "1979a"),
    ("Plange_1979b", ["Plange"], "1979b"),
    ("deHaas_2017", ["de Haas"], "2017"),
    ("vanderGeest_2011", ["van der Geest"], "2011"),
    ("vanHear_1984", ["Van Hear"], "1984"),
    ("Frankema&vanWaijenburg_2012", ["Frankema", "van Waijenburg"], "2012"),
    ("MeierzuSelhausen&Weisdorf_2016", ["Meier zu Selhausen", "Weisdorf"], "2016"),
    ("Canagarajah&Portner_2003", ["Canagarajah", "Pörtner"], "2003"),
    ("Ronnback_2014", ["Rönnbäck"], "2014"),
    ("AryeeteyAttoh&Chatterjee_1988", ["Aryeetey-Attoh", "Chatterjee"], "1988"),
    ("AppiahKubiEtAl_2008", ["Appiah-Kubi", "Oduro", "Senadza"], "2008"),
    ("ORourke_2001", ["O'Rourke"], "2001"),
    ("FAO_2001", ["Food and Agriculture Organization"], "2001"),
    ("Brukum_1998", ["Brukum"], "1998"),
]


def _rows():
    return [
        {
            "doc_id": doc_id,
            "year": year,
            "display_label": build_display_label(surnames, year),
            "first_author_surname": surnames[0],
        }
        for doc_id, surnames, year in WORKS
    ]


class BuildDisplayLabelTests(unittest.TestCase):
    def test_label_forms(self):
        self.assertEqual("World Bank (1995)", build_display_label(["World Bank"], "1995"))
        self.assertEqual(
            "Ghana Statistical Service (2000a)",
            build_display_label(["Ghana Statistical Service"], "2000a"),
        )
        self.assertEqual("de Haas (2017)", build_display_label(["de Haas"], 2017))
        self.assertEqual("van der Geest (2011)", build_display_label(["van der Geest"], "2011"))
        self.assertEqual(
            "Canagarajah and Pörtner (2003)",
            build_display_label(["Canagarajah", "Pörtner"], "2003"),
        )
        self.assertEqual(
            "Appiah-Kubi et al. (2008)",
            build_display_label(["Appiah-Kubi", "Oduro", "Senadza"], "2008"),
        )
        self.assertEqual(
            "Meier zu Selhausen and Weisdorf (2016)",
            build_display_label(["Meier zu Selhausen", "Weisdorf"], "2016"),
        )

    def test_typographic_marks_become_plain_ones(self):
        self.assertEqual(
            "ARYEETEY-ATTOH and CHATTERJEE (1988)",
            build_display_label(["ARYEETEY‐ATTOH", "CHATTERJEE"], "1988"),
        )
        self.assertEqual("O'Rourke (2001)", build_display_label(["O’Rourke"], "2001"))

    def test_characters_that_would_break_a_citation_are_dropped(self):
        self.assertEqual(
            "Institute of Statistics ISSER (1999)",
            build_display_label(["Institute of Statistics (ISSER);"], "1999"),
        )

    def test_no_label_without_a_surname_or_a_year(self):
        self.assertEqual("", build_display_label([], "1999"))
        self.assertEqual("", build_display_label(["Plange", ""], "1979"))
        self.assertEqual("", build_display_label(["Plange"], ""))
        self.assertEqual("", build_display_label(["Plange"], "n.d."))


class LabelRoundTripTests(unittest.TestCase):
    def setUp(self):
        render.set_metadata_labels(_rows())
        self.doc_ids = [doc_id for doc_id, _, _ in WORKS]
        self.lookup = render._build_display_lookup(self.doc_ids)

    def tearDown(self):
        render.set_metadata_labels([])

    def test_the_catalogue_check_finds_no_problem(self):
        self.assertEqual(
            {"checked": len(WORKS), "problems": []}, render.check_citation_labels(_rows())
        )

    def test_every_generated_label_comes_back_in_every_citation_form(self):
        for doc_id in self.doc_ids:
            narrative = render.render_citation(doc_id, 3)
            label, _, tail = narrative.rpartition(" (")
            texts = [
                f"As {narrative} shows, this holds.",
                f"This holds ({label} {tail}.",
                f"This holds ({label}, {tail}.",
                f"This holds (Austin 2008, p.1; {label} {tail}.",
                f"{label} ({tail[:-1]}; Austin 2008, p.1) agree.",
            ]
            if not re.fullmatch(render._DISPLAY_LABEL, label):
                texts.append(f"As {narrative.lower()} shows, this holds.")
            for text in texts:
                found = [
                    citation["doc_id"]
                    for citation in render.parse_citations(text, display_lookup=self.lookup)
                    if citation["label"] != "Austin"
                ]
                self.assertEqual([doc_id], found, text)

    def test_the_writer_renders_and_releases_these_labels(self):
        evidence = {
            "E0001": {"doc_id": "WorldBank_1995", "page": 3},
            "E0002": {"doc_id": "GhanaStatisticalService_2000a", "page": 5},
            "E0003": {"doc_id": "GhanaStatisticalService_2000b", "page": 9},
            "E0004": {"doc_id": "Canagarajah&Portner_2003", "page": 12},
            "E0005": {"doc_id": "MeierzuSelhausen&Weisdorf_2016", "page": 2},
        }
        draft = (
            "The World Bank [E0001] counted the poor by region. Poverty fell "
            "in the forest belt [E0002] and the method changed [E0003]. "
            "Canagarajah and Pörtner [E0004] date the divergence, and mission "
            "records extend it backwards [E0005]."
        )
        text, stats = writer._render_evidence_id_citations(draft, evidence)
        self.assertEqual(5, stats["replacements"])
        self.assertEqual(0, stats["attribution_mismatches"])
        self.assertIn("The World Bank (1995, p.3) counted", text)
        self.assertIn("(Ghana Statistical Service 2000a, p.5)", text)
        self.assertIn("(Ghana Statistical Service 2000b, p.9)", text)
        self.assertIn("Canagarajah and Pörtner (2003, p.12)", text)
        self.assertIn("(Meier zu Selhausen and Weisdorf 2016, p.2)", text)

        docs = {item["doc_id"] for item in evidence.values()}
        pairs = {(item["doc_id"], item["page"]) for item in evidence.values()}
        lookup = render._build_display_lookup(docs)
        cleaned, removed = writer._remove_invalid_citations(
            text, docs, pairs, display_lookup=lookup,
        )
        self.assertEqual([], removed)
        self.assertEqual(text, cleaned)
        self.assertEqual(
            [
                "WorldBank_1995", "GhanaStatisticalService_2000a",
                "GhanaStatisticalService_2000b", "Canagarajah&Portner_2003",
                "MeierzuSelhausen&Weisdorf_2016",
            ],
            [c["doc_id"] for c in render.parse_citations(cleaned, display_lookup=lookup)],
        )

    def test_part_of_a_label_is_not_the_label(self):
        for text in (
            "Poverty fell (Statistical Service 2000a, p.5).",
            "Service (2000a, p.5) reports a fall in poverty.",
            "The Bank (1995, p.3) counted the poor.",
        ):
            credited = [
                citation["doc_id"]
                for citation in render.parse_citations(text, display_lookup=self.lookup)
            ]
            self.assertNotIn("GhanaStatisticalService_2000a", credited, text)
            self.assertNotIn("WorldBank_1995", credited, text)
        _, removed = writer._remove_invalid_citations(
            "The Bank (1995, p.3) counted the poor.", set(self.doc_ids), None,
            display_lookup=self.lookup,
        )
        self.assertEqual(["unresolved_display"], [item["reason"] for item in removed])

    def test_a_catalogue_whose_labels_fit_the_grammar_adds_nothing(self):
        render.set_metadata_labels([
            {"doc_id": "North_1989", "year": 1989},
            {"doc_id": "Frankema&vanWaijenburg_2012", "year": 2012},
            {"doc_id": "AcemogluEtAl_2001", "year": 2001},
        ])
        self.assertEqual("", render._KNOWN_LABELS)
        plain = render.compile_citation_patterns("")
        self.assertEqual(plain["narrative"].pattern, render.DISPLAY_CITE_RE.pattern)
        self.assertEqual(plain["paren"].pattern, render.DISPLAY_PAREN_CITE_RE.pattern)

    def test_only_the_labels_outside_the_grammar_are_listed(self):
        known = render.known_label_alternation(
            ["North", "van der Geest", "World Bank", "Canagarajah and Pörtner",
             "Appiah-Kubi et al."]
        )
        self.assertEqual(r"Canagarajah\s+and\s+Pörtner|World\s+Bank", known)


class IngestWritesTheLabelTests(unittest.TestCase):
    def test_the_catalogue_carries_the_bibliographic_label(self):
        works = [
            ("World Bank (1995).pdf", "WB95", "{World Bank}", "1995",
             "Ghana: Poverty Past, Present and Future"),
            ("de Haas (2017).pdf", "dH17", "de Haas, Michiel", "2017",
             "Measuring Rural Welfare in Colonial Africa"),
            ("Canagarajah and Pörtner (2003).pdf", "CP03",
             "Canagarajah, Sudharshan and P{\\\"o}rtner, Claus C.", "2003",
             "Evolution of Poverty and Welfare in Ghana in the 1990s"),
            ("Plange (1979a).pdf", "Pa", "Plange, Nii-K.", "1979",
             "Underdevelopment in Northern Ghana"),
            ("Plange (1979b).pdf", "Pb", "Plange, Nii-K.", "1979",
             "Opportunity Cost and Labour Migration"),
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            corpus = Path(temp_dir) / "papers"
            entries = []
            for file_name, key, author, year, title in works:
                make_pdf(corpus / file_name, title, "Synthetic test paper", "n.d.")
                entries.append(("article", key, {
                    "author": author, "title": title, "year": year, "file": file_name,
                }))
            write_bib(corpus / "bibliography.bib", entries)
            output = Path(temp_dir) / "metadata.csv"
            rows = ingest_corpus(
                corpus, output_csv=output, sidecar_bib=corpus / "bibliography.bib",
                use_llm=False, use_crossref=False, use_openalex=False, progress=False,
            )
            with output.open(encoding="utf-8", newline="") as handle:
                catalogue = {row["doc_id"]: row for row in csv.DictReader(handle)}
        labels = {row.doc_id: row.display_label for row in rows}
        self.assertEqual(
            {
                "WorldBank_1995": "World Bank (1995)",
                "deHaas_2017": "de Haas (2017)",
                "Canagarajah&Portner_2003": "Canagarajah and Pörtner (2003)",
                "Plange_1979a": "Plange (1979a)",
                "Plange_1979b": "Plange (1979b)",
            },
            labels,
        )
        self.assertEqual(labels, {k: v["display_label"] for k, v in catalogue.items()})
        problems = render.check_citation_labels(list(catalogue.values()))["problems"]
        self.assertEqual([], problems)


class CheckerReadsTheLabelsTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.metadata = root / "metadata.csv"
        self.data = root / "data"
        (self.data / "page_text").mkdir(parents=True)
        with self.metadata.open("w", encoding="utf-8", newline="") as handle:
            out = csv.DictWriter(
                handle, fieldnames=["doc_id", "display_label", "first_author_surname", "year"],
            )
            out.writeheader()
            for row in _rows():
                out.writerow(row)
        for doc_id, _, _ in WORKS:
            (self.data / f"{doc_id}.json").write_text(
                json.dumps({"doc_id": doc_id, "page_count": 40}), encoding="utf-8",
            )

    def tearDown(self):
        self.temp_dir.cleanup()

    def _check(self, text):
        return CHECKER.check_review(text, str(self.metadata), str(self.data), quote_check=False)

    def test_the_checker_resolves_what_the_writer_releases(self):
        review = (
            "The World Bank (1995, p.3) counted the poor by region. Poverty "
            "fell in the forest belt (Ghana Statistical Service 2000a, p.5) and "
            "the method changed (Ghana Statistical Service 2000b, p.9). Plange "
            "(1979a, p.2) dates the labour reserve, and a second paper disputes "
            "the opportunity-cost reading (Plange 1979b, p.4). Canagarajah and "
            "Pörtner (2003, p.12) date the divergence (Meier zu Selhausen and "
            "Weisdorf 2016, p.2; van der Geest 2011, p.7)."
        )
        result = self._check(review)
        self.assertEqual(0, result["e1"], result["e1_details"])
        self.assertEqual(0, result["e2"])
        self.assertEqual(0, result["e3"], result["e3_details"])
        self.assertEqual(8, result["n_citations"])
        self.assertEqual(
            {
                "WorldBank_1995", "GhanaStatisticalService_2000a",
                "GhanaStatisticalService_2000b", "Plange_1979a", "Plange_1979b",
                "Canagarajah&Portner_2003", "MeierzuSelhausen&Weisdorf_2016",
                "vanderGeest_2011",
            },
            set(result["docs_cited"]),
        )

    def test_a_wrong_or_missing_letter_is_out_of_corpus(self):
        self.assertEqual(1, self._check("Plange (1979c, p.2) dates the reserve.")["e1"])
        self.assertEqual(1, self._check("Plange (1979, p.2) dates the reserve.")["e1"])
        self.assertEqual(0, self._check("Plange (1979b, p.2) dates the reserve.")["e1"])

    def test_part_of_a_label_credits_no_document(self):
        result = self._check("The Bank (1995, p.3) counted the poor.")
        self.assertEqual(1, result["e1"])
        self.assertEqual([], list(result["docs_cited"]))
        result = self._check("Poverty fell (Statistical Service 2000a, p.5).")
        self.assertEqual([], list(result["docs_cited"]))

    def test_the_checker_leaves_its_patterns_as_it_found_them(self):
        before = CHECKER._GROUP_DISPLAY_UNIT_RE.pattern
        self._check("The World Bank (1995, p.3) counted the poor.")
        self.assertEqual(before, CHECKER._GROUP_DISPLAY_UNIT_RE.pattern)
        self.assertEqual(CHECKER._GROUP_DISPLAY_LABEL_RE, CHECKER._GROUP_LABEL)


if __name__ == "__main__":
    unittest.main()
