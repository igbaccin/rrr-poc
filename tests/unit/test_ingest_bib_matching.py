import tempfile
import unicodedata
import unittest
from pathlib import Path

from _pdf_fixtures import FILLER, make_pdf, write_bib

from rrr.ingest import (
    ExtractedMeta,
    _bib_file_names,
    _fuzzy_title_match,
    hold_repeated_bib_entries,
    ingest_corpus,
    match_bib_entry,
    match_bib_entry_detailed,
    parse_bib_sidecar,
    parse_filename_heuristic,
    parse_filename_readings,
)


def _entry(key, surnames, year, title="An Untitled Study of Something Else", files=()):
    surnames = list(surnames)
    return {
        "key": key, "title": title, "year": year, "venue": "", "volume": "",
        "number": "", "pages": "", "doi": "",
        "authors_short": " & ".join(surnames),
        "author_full": "; ".join(surnames),
        "first_surname": surnames[0] if surnames else "",
        "surnames": surnames,
        "files": list(files),
    }


def _match(file_name, sidecar, text=""):
    return match_bib_entry_detailed(Path("corpus") / file_name, text, sidecar)


class BibFileFieldTests(unittest.TestCase):
    def test_file_field_forms(self):
        self.assertEqual(["Plange (1979a).pdf"], _bib_file_names("Plange (1979a).pdf"))
        self.assertEqual(
            ["de Haas (2017).pdf"],
            _bib_file_names(r":C\:\\Users\\me\\Papers\\de Haas (2017).pdf:PDF"),
        )
        self.assertEqual(
            ["de Haas (2017).pdf"], _bib_file_names(":papers/de Haas (2017).pdf:PDF"),
        )
        self.assertEqual(
            ["Moradi - 2012 - Climate.pdf"],
            _bib_file_names(
                "Full Text PDF:files/412/Moradi - 2012 - Climate.pdf:application/pdf"
            ),
        )
        self.assertEqual(
            ["World Bank (1995).pdf"],
            _bib_file_names(r"C:\Users\me\Corpus\World Bank (1995).pdf"),
        )
        self.assertEqual(
            ["A (2001).pdf", "B (2002).PDF"],
            _bib_file_names(
                "Snapshot:files/1/page.html:text/html;:x/A (2001).pdf:PDF;"
                "Scan:y/B (2002).PDF:application/pdf"
            ),
        )
        self.assertEqual([], _bib_file_names(""))
        self.assertEqual([], _bib_file_names("notes:files/3/notes.txt:text/plain"))

    def test_parse_bib_sidecar_keeps_the_file_field(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            bib = write_bib(Path(temp_dir) / "bibliography.bib", [
                ("article", "PlangeA", {
                    "author": "Plange, Nii-K.", "title": "A", "year": "1979",
                    "file": "Plange (1979a).pdf",
                }),
                ("article", "Brukum1998", {
                    "author": "Brukum, N. J. K.", "title": "B", "year": "1998",
                }),
            ])
            entries = {entry["key"]: entry for entry in parse_bib_sidecar(bib)}
        self.assertEqual(["Plange (1979a).pdf"], entries["PlangeA"]["files"])
        self.assertEqual([], entries["Brukum1998"]["files"])

    def test_the_file_field_is_matched_first_and_without_regard_to_case(self):
        sidecar = [
            _entry("Wrong1979", ["Plange"], "1979", files=[]),
            _entry("Right", ["Somebody"], "2001", files=["PLANGE (1979A).PDF"]),
        ]
        match = _match("Plange (1979a).pdf", sidecar)
        self.assertEqual("Right", match.entry["key"])
        self.assertEqual("file_field", match.route)

    def test_the_file_field_matches_across_unicode_forms(self):
        composed = "Canagarajah and Pörtner (2003).pdf"
        decomposed = unicodedata.normalize("NFD", composed)
        self.assertNotEqual(composed, decomposed)
        sidecar = [_entry("CP2003", ["Canagarajah", "Pörtner"], "2003", files=[composed])]
        self.assertEqual("file_field", _match(decomposed, sidecar).route)

    def test_a_file_named_by_two_entries_is_not_given_to_either_by_that_route(self):
        sidecar = [
            _entry("One", ["Alpha"], "2001", files=["shared.pdf"]),
            _entry("Two", ["Beta"], "2002", files=["shared.pdf"]),
        ]
        match = _match("shared.pdf", sidecar)
        self.assertIsNone(match.entry)
        self.assertIn("One, Two all name this file", match.note)


class FileNameFormTests(unittest.TestCase):
    def _read(self, stem):
        return parse_filename_heuristic(Path(stem + ".pdf"))

    def test_accepted_forms(self):
        cases = {
            "Plange (1979)": (["Plange"], "1979", "", False),
            "Plange 1979": (["Plange"], "1979", "", False),
            "Plange-1979": (["Plange"], "1979", "", False),
            "Plange_1979b": (["Plange"], "1979", "b", False),
            "Plange (1979a)": (["Plange"], "1979", "a", False),
            "Aryeetey and Kanbur (2008)": (["Aryeetey", "Kanbur"], "2008", "", False),
            "Aryeetey & Kanbur (2008)": (["Aryeetey", "Kanbur"], "2008", "", False),
            "Appiah-Kubi et al. (2008)": (["Appiah-Kubi"], "2008", "", True),
            "Appiah-Kubi et al (2008)": (["Appiah-Kubi"], "2008", "", True),
            "Aryeetey-Attoh and Chatterjee (1988)": (
                ["Aryeetey-Attoh", "Chatterjee"], "1988", "", False),
            "de Haas (2017)": (["de Haas"], "2017", "", False),
            "Canagarajah and Pörtner (2003)": (["Canagarajah", "Pörtner"], "2003", "", False),
            "Ghana Statistical Service (2000a)": (
                ["Ghana Statistical Service"], "2000", "a", False),
            "World Bank (1995)": (["World Bank"], "1995", "", False),
            "Frankema and van Waijenburg (2012)": (
                ["Frankema", "van Waijenburg"], "2012", "", False),
            "AcemogluEtAl_2001": (["Acemoglu"], "2001", "", True),
            "North&Weingast_1989": (["North", "Weingast"], "1989", "", False),
        }
        for stem, (surnames, year, letter, et_al) in cases.items():
            parsed = self._read(stem)
            self.assertIsNotNone(parsed, stem)
            self.assertEqual(surnames, parsed["surnames"], stem)
            self.assertEqual(year, parsed["year"], stem)
            self.assertEqual(letter, parsed["letter"], stem)
            self.assertEqual(et_al, parsed["et_al"], stem)

    def test_reference_manager_and_other_common_forms(self):
        cases = {
            "Moradi - 2012 - Climate, height and economic development": (["Moradi"], "2012", False),
            "Moradi et al. - 2013 - Heights and Development": (["Moradi"], "2013", True),
            "Aboagye and Bolt - 2021 - Long-term trends": (["Aboagye", "Bolt"], "2021", False),
            "Frankema and van Waijenburg – 2012 – Structural impediments": (
                ["Frankema", "van Waijenburg"], "2012", False),
            "Plange (1979) Underdevelopment in Northern Ghana": (["Plange"], "1979", False),
            "World Bank (1995) - Ghana Poverty Past Present and Future": (
                ["World Bank"], "1995", False),
            "Smith2019": (["Smith"], "2019", False),
            "2012 - Moradi - Climate": (["Moradi"], "2012", False),
            "2012_Moradi_Climate_height": (["Moradi"], "2012", False),
            "2012 Moradi": (["Moradi"], "2012", False),
        }
        for stem, (surnames, year, et_al) in cases.items():
            parsed = self._read(stem)
            self.assertIsNotNone(parsed, stem)
            self.assertEqual((surnames, year, et_al),
                             (parsed["surnames"], parsed["year"], parsed["et_al"]), stem)

    def test_a_year_inside_the_title_gives_a_second_reading_only(self):
        readings = parse_filename_readings(
            Path("Moradi - 2012 - Heights in Ghana 1870-1980.pdf")
        )
        self.assertEqual("2012", readings[0]["year"])
        self.assertEqual(["Moradi"], readings[0]["surnames"])
        self.assertEqual(["2012", "1980"], [reading["year"] for reading in readings])

    def test_names_that_only_look_like_one_are_not_read(self):
        for stem in ("Report on Ghana 1957 census", "10.1016_j.eeh.2021.101405",
                     "1-s2.0-S0014498321000280-main", "chapter 3 (draft)"):
            self.assertEqual([], parse_filename_readings(Path(stem + ".pdf")), stem)

    def test_the_older_convention_keeps_its_name_field(self):
        self.assertEqual("AcemogluEtAl", self._read("AcemogluEtAl_2001")["name"])
        self.assertEqual("North&Weingast", self._read("North&Weingast_1989")["name"])

    def test_names_without_author_and_year_are_not_read(self):
        for stem in ("notes", "IMG_20190101", "1979", "(1979)", "scan 12", "chapter-3"):
            self.assertIsNone(self._read(stem), stem)

    def test_a_surname_ending_in_etal_is_not_cut(self):
        parsed = self._read("Metal_2001")
        self.assertEqual(["Metal"], parsed["surnames"])
        self.assertFalse(parsed["et_al"])


class FileNameMatchTests(unittest.TestCase):
    SIDECAR = [
        _entry("AryeeteyAttohChatterjee_1988", ["Aryeetey-Attoh", "Chatterjee"], "1988"),
        _entry("deHaas2017", ["de Haas"], "2017"),
        _entry("CanagarajahPortner2003", ["Canagarajah", "Pörtner"], "2003"),
        _entry("GSS2000a", ["Ghana Statistical Service"], "2000"),
        _entry("GSS2000b", ["Ghana Statistical Service"], "2000"),
        _entry("WorldBank1995", ["World Bank"], "1995"),
        _entry("WorldBank2011", ["World Bank"], "2011"),
        _entry("FrankemaWaijenburg2012", ["Frankema", "van Waijenburg"], "2012"),
        _entry("AryeeteyKanbur2008", ["Aryeetey", "Kanbur"], "2008"),
        _entry("AryeeteyMcKay2008", ["Aryeetey", "McKay"], "2008"),
        _entry("AppiahKubi2008solo", ["Appiah-Kubi"], "2008"),
        _entry("AppiahKubi2008", ["Appiah-Kubi", "Oduro", "Senadza"], "2008"),
        _entry("FAO2001", ["Food and Agriculture Organization"], "2001"),
    ]

    def test_surname_year_file_names_find_their_entry(self):
        expected = {
            "Aryeetey-Attoh and Chatterjee (1988).pdf": "AryeeteyAttohChatterjee_1988",
            "de Haas (2017).pdf": "deHaas2017",
            "Haas (2017).pdf": "deHaas2017",
            "Canagarajah and Pörtner (2003).pdf": "CanagarajahPortner2003",
            "Canagarajah and Portner (2003).pdf": "CanagarajahPortner2003",
            "Ghana Statistical Service (2000a).pdf": "GSS2000a",
            "Ghana Statistical Service (2000b).pdf": "GSS2000b",
            "World Bank (1995).pdf": "WorldBank1995",
            "World Bank 2011.pdf": "WorldBank2011",
            "Frankema and van Waijenburg (2012).pdf": "FrankemaWaijenburg2012",
            "Aryeetey and Kanbur (2008).pdf": "AryeeteyKanbur2008",
            "Aryeetey & McKay (2008).pdf": "AryeeteyMcKay2008",
            "Appiah-Kubi et al. (2008).pdf": "AppiahKubi2008",
            "Appiah-Kubi (2008).pdf": "AppiahKubi2008solo",
            "Food and Agriculture Organization (2001).pdf": "FAO2001",
        }
        for file_name, key in expected.items():
            match = _match(file_name, self.SIDECAR)
            self.assertIsNotNone(match.entry, f"{file_name}: {match.note}")
            self.assertEqual(key, match.entry["key"], file_name)
            self.assertEqual("file_name", match.route, file_name)

    def test_reference_manager_file_names_find_their_entry(self):
        sidecar = self.SIDECAR + [
            _entry("MoradiClimate", ["Moradi"], "2012"),
            _entry("Moradi1980", ["Moradi"], "1980"),
            _entry("MoradiAustinBaten2013", ["Moradi", "Austin", "Baten"], "2013"),
        ]
        expected = {
            "Moradi - 2012 - Heights in Ghana 1870-1980.pdf": "MoradiClimate",
            "Moradi et al. - 2013 - Heights and Development in a Cash-Crop Colony.pdf":
                "MoradiAustinBaten2013",
            "Aryeetey and Kanbur - 2008 - Ghana's economy at half century.pdf":
                "AryeeteyKanbur2008",
            "World Bank (1995) Ghana Poverty Past Present and Future.pdf": "WorldBank1995",
            "2017 - de Haas - Measuring rural welfare.pdf": "deHaas2017",
            "2003_Canagarajah and Pörtner_Evolution of poverty.pdf": "CanagarajahPortner2003",
            "Moradi2012.pdf": "MoradiClimate",
        }
        for file_name, key in expected.items():
            match = _match(file_name, sidecar)
            self.assertIsNotNone(match.entry, f"{file_name}: {match.note}")
            self.assertEqual((key, "file_name"), (match.entry["key"], match.route), file_name)

    def test_a_later_reading_is_used_when_the_first_finds_no_entry(self):
        sidecar = [_entry("Moradi1980", ["Moradi"], "1980")]
        text = "Alexander Moradi. Heights in Ghana. 1980. " + FILLER
        match = _match("Moradi - 2012 - Heights in Ghana 1870-1980.pdf", sidecar, text)
        self.assertEqual("Moradi1980", match.entry["key"])
        self.assertIsNone(
            _match("Moradi - 2012 - Heights in Ghana 1870-1980.pdf", sidecar, FILLER).entry
        )

    def test_the_key_route_still_works(self):
        match = _match("worldbank1995.pdf", self.SIDECAR)
        self.assertEqual(("WorldBank1995", "key"), (match.entry["key"], match.route))

    def test_a_year_letter_in_the_bibliography_year_decides(self):
        sidecar = [
            _entry("PlangeOne", ["Plange"], "1979a"),
            _entry("PlangeTwo", ["Plange"], "1979b"),
        ]
        self.assertEqual("PlangeOne", _match("Plange (1979a).pdf", sidecar).entry["key"])
        self.assertEqual("PlangeTwo", _match("Plange_1979b.pdf", sidecar).entry["key"])

    def test_the_title_decides_when_no_letter_does(self):
        sidecar = [
            _entry("PlangeOne", ["Plange"], "1979",
                   title="Underdevelopment in Northern Ghana: Natural Causes or Colonial Capitalism"),
            _entry("PlangeTwo", ["Plange"], "1979",
                   title="Opportunity Cost and Labour Migration: A Misinterpretation"),
        ]
        text = "Opportunity cost and labour migration: a misinterpretation. Nii-K. Plange 1979. " + FILLER
        match = _match("Plange (1979).pdf", sidecar, text)
        self.assertEqual("PlangeTwo", match.entry["key"])

    def test_a_pdf_that_fits_two_entries_is_reported_and_never_guessed(self):
        sidecar = [
            _entry("PlangeOne", ["Plange"], "1979"),
            _entry("PlangeTwo", ["Plange"], "1979"),
        ]
        match = _match("Plange (1979).pdf", sidecar, "Plange 1979 " + FILLER)
        self.assertIsNone(match.entry)
        self.assertTrue(match.ambiguous)
        self.assertIn("PlangeOne, PlangeTwo", match.note)
        self.assertIn("2 entries fit the file name", match.note)
        self.assertIsNone(match_bib_entry(Path("Plange (1979).pdf"), "Plange 1979", sidecar))

    def test_a_loose_fit_needs_the_author_on_the_first_pages(self):
        sidecar = [_entry("Li2010", ["Li"], "2010")]
        lindert = "Peter H. Lindert. Growing Public. 2010. " + FILLER
        self.assertIsNone(_match("Lindert_2010.pdf", sidecar, lindert).entry)
        sidecar = [_entry("Acemoglu2001", ["Acemoglu", "Johnson", "Robinson"], "2001")]
        text = "Daron Acemoglu, Simon Johnson and James Robinson. 2001. " + FILLER
        self.assertEqual(
            "Acemoglu2001", _match("AcemogluJohnsonRobinson_2001.pdf", sidecar, text).entry["key"]
        )

    def test_a_loose_fit_is_accepted_when_the_year_letters_agree(self):
        sidecar = [
            _entry("PlangeA_1979a", ["PlangeA"], "1979a"),
            _entry("PlangeB_1979b", ["PlangeB"], "1979b"),
        ]
        text = "Nii-K. Plange. 1979. " + FILLER
        self.assertEqual(
            "PlangeB_1979b", _match("Plange (1979b).pdf", sidecar, text).entry["key"]
        )
        self.assertIsNone(_match("Plange (1979).pdf", sidecar, text).entry)

    def test_the_note_says_why_nothing_matched(self):
        match = _match("Brukum (1998).pdf", self.SIDECAR, FILLER)
        self.assertIsNone(match.entry)
        self.assertFalse(match.ambiguous)
        self.assertIn("file name read as Brukum (1998)", match.note)
        self.assertIn("no bibliography entry has that first author and year", match.note)
        self.assertIn("no bibliography title found on the first pages", match.note)
        match = _match("scan0042.pdf", self.SIDECAR, FILLER)
        self.assertIn("file name has no author-and-year form", match.note)


class TitleMatchTests(unittest.TestCase):
    TITLE = "Evolution of Poverty and Welfare in Ghana in the 1990s: Achievements and Challenges"

    def test_ligatures_and_accents_are_folded_on_both_sides(self):
        title = "Efficient Offices and the Financing of Pörtner's Welfare Reforms"
        page = (
            "E\ufb03cient O\ufb03ces and the \ufb01nancing of Portner\u2019s welfare reforms\n"
            "Claus C. Portner, 2003. " + FILLER
        )
        self.assertTrue(_fuzzy_title_match(title, page))

    def test_punctuation_between_title_words_does_not_hide_the_title(self):
        page = "GHANA: POVERTY PAST, PRESENT AND FUTURE\nWorld Bank 1995\n" + FILLER
        self.assertTrue(_fuzzy_title_match("Ghana: Poverty Past, Present and Future", page))

    def test_the_match_stays_strict(self):
        scattered = (
            "Welfare rose. Poverty fell in Ghana. The 1990s brought challenges and "
            "achievements, and the evolution of prices mattered. " + FILLER
        )
        self.assertFalse(_fuzzy_title_match(self.TITLE, scattered))
        other = "Poverty and Welfare in Uganda in the 1980s: Achievements\n" + FILLER
        self.assertFalse(_fuzzy_title_match(self.TITLE, other))
        self.assertFalse(_fuzzy_title_match("Ghana in 1957", "Ghana in 1957 " + FILLER))

    def test_the_title_route_needs_author_and_year_on_the_page(self):
        sidecar = [_entry("CP2003", ["Canagarajah", "Pörtner"], "2003", title=self.TITLE)]
        page = self.TITLE + "\nSudharshan Canagarajah and Claus C. Pörtner\n2003\n" + FILLER
        match = _match("wp61.pdf", sidecar, page)
        self.assertEqual(("CP2003", "title"), (match.entry["key"], match.route))
        without_year = page.replace("2003", "")
        self.assertIsNone(_match("wp61.pdf", sidecar, without_year).entry)

    def test_a_lettered_bibliography_year_still_matches_by_title(self):
        sidecar = [_entry("PlangeOne", ["Plange"], "1979a", title=self.TITLE)]
        page = self.TITLE + "\nNii-K. Plange\n1979\n" + FILLER
        self.assertEqual("PlangeOne", _match("x.pdf", sidecar, page).entry["key"])


class CascadeMatchingTests(unittest.TestCase):
    WORKS = [
        ("Plange (1979a).pdf", "PlangeUnder", "Plange, Nii-K.", "1979",
         "Underdevelopment in Northern Ghana: Natural Causes or Colonial Capitalism"),
        ("Plange (1979b).pdf", "PlangeOpportunity", "Plange, Nii-K.", "1979",
         "Opportunity Cost and Labour Migration: A Misinterpretation of Proletarianisation"),
        ("de Haas (2017).pdf", "deHaas2017", "de Haas, Michiel", "2017",
         "Measuring Rural Welfare in Colonial Africa: Did Uganda's Smallholders Thrive"),
        ("World Bank (1995).pdf", "WorldBank1995", "{World Bank}", "1995",
         "Ghana: Poverty Past, Present and Future"),
        ("Canagarajah and Pörtner (2003).pdf", "CP2003",
         "Canagarajah, Sudharshan and P{\\\"o}rtner, Claus C.", "2003",
         "Evolution of Poverty and Welfare in Ghana in the 1990s"),
        ("Aryeetey-Attoh and Chatterjee (1988).pdf", "AAC1988",
         "Aryeetey-Attoh, Samuel and Chatterjee, Lata", "1988",
         "Regional Inequalities in Ghana: Assessment and Policy Issues"),
    ]

    def _corpus(self, root, with_file_field, text_matches_title=True):
        corpus = root / "papers"
        entries = []
        for file_name, key, author, year, title in self.WORKS:
            shown = title if text_matches_title else "Untitled manuscript, cover page missing"
            make_pdf(corpus / file_name, shown, "Prepared for the unit tests", "n.d.")
            fields = {"author": author, "title": title, "year": year}
            if with_file_field:
                fields["file"] = file_name
            entries.append(("article", key, fields))
        write_bib(corpus / "bibliography.bib", entries)
        return corpus

    def _ingest(self, corpus):
        return ingest_corpus(
            corpus, sidecar_bib=corpus / "bibliography.bib",
            use_llm=False, use_crossref=False, use_openalex=False, progress=False,
        )

    def test_the_file_field_matches_every_pdf_even_when_the_text_gives_nothing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            rows = self._ingest(self._corpus(Path(temp_dir), True, text_matches_title=False))
        self.assertEqual(len(self.WORKS), len(rows))
        for row in rows:
            self.assertEqual(("bib_sidecar", "high"), (row.source, row.confidence), row.pdf_path)
            self.assertEqual("bib_match:file_field", row.notes)
        ids = {Path(row.pdf_path).name: row.doc_id for row in rows}
        self.assertEqual("Plange_1979a", ids["Plange (1979a).pdf"])
        self.assertEqual("Plange_1979b", ids["Plange (1979b).pdf"])
        self.assertEqual("deHaas_2017", ids["de Haas (2017).pdf"])
        self.assertEqual("WorldBank_1995", ids["World Bank (1995).pdf"])
        self.assertEqual("Canagarajah&Portner_2003", ids["Canagarajah and Pörtner (2003).pdf"])
        self.assertEqual(
            "AryeeteyAttoh&Chatterjee_1988", ids["Aryeetey-Attoh and Chatterjee (1988).pdf"]
        )

    def test_file_names_alone_match_every_pdf_that_has_one_candidate(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            rows = self._ingest(self._corpus(Path(temp_dir), False, text_matches_title=False))
        by_name = {Path(row.pdf_path).name: row for row in rows}
        for file_name in (
            "de Haas (2017).pdf", "World Bank (1995).pdf",
            "Canagarajah and Pörtner (2003).pdf",
            "Aryeetey-Attoh and Chatterjee (1988).pdf",
        ):
            self.assertEqual("bib_match:file_name", by_name[file_name].notes, file_name)
            self.assertEqual("high", by_name[file_name].confidence)
        for file_name in ("Plange (1979a).pdf", "Plange (1979b).pdf"):
            row = by_name[file_name]
            self.assertEqual(("bib_ambiguous", "low", ""), (row.source, row.confidence, row.doc_id))
            self.assertIn("PlangeUnder", row.notes)
            self.assertIn("PlangeOpportunity", row.notes)

    def test_the_title_settles_the_pair_when_the_pages_carry_it(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            rows = self._ingest(self._corpus(Path(temp_dir), False, text_matches_title=True))
        by_name = {Path(row.pdf_path).name: row for row in rows}
        self.assertEqual("PlangeUnder", by_name["Plange (1979a).pdf"].bib_key)
        self.assertEqual("PlangeOpportunity", by_name["Plange (1979b).pdf"].bib_key)
        self.assertEqual("Plange_1979a", by_name["Plange (1979a).pdf"].doc_id)
        self.assertEqual("Plange_1979b", by_name["Plange (1979b).pdf"].doc_id)

    def test_an_unmatched_pdf_says_why(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            corpus = self._corpus(Path(temp_dir), True)
            make_pdf(corpus / "scan0042.pdf", "A Paper Nobody Listed", "A. N. Other", "1961")
            rows = self._ingest(corpus)
        row = next(row for row in rows if row.pdf_path.endswith("scan0042.pdf"))
        self.assertEqual(("none", "failed"), (row.source, row.confidence))
        self.assertIn("unmatched: bibliography: file name has no author-and-year form", row.notes)
        self.assertIn("no bibliography title found on the first pages", row.notes)
        self.assertIn("no DOI on the first pages", row.notes)
        self.assertIn("model extraction not requested", row.notes)


class RepeatedEntryTests(unittest.TestCase):
    def _row(self, pdf, route, sha):
        return ExtractedMeta(
            doc_id="Plange_1979", year="1979", pdf_path=str(Path("corpus") / pdf),
            bib_key="PlangeOne", content_sha1=sha, source="bib_sidecar",
            confidence="high", notes=f"bib_match:{route}",
        )

    def test_the_second_pdf_of_one_entry_waits_for_the_user(self):
        rows = [
            self._row("a copy.pdf", "title", "111"),
            self._row("Plange (1979).pdf", "file_name", "222"),
        ]
        held = hold_repeated_bib_entries(rows)
        self.assertEqual(["a copy.pdf"], [Path(row.pdf_path).name for row in held])
        copy = held[0]
        self.assertEqual(("", "low", "bib_repeated"), (copy.doc_id, copy.confidence, copy.source))
        self.assertIn("PlangeOne is already matched to Plange (1979).pdf", copy.notes)
        self.assertIn("give this PDF its own entry", copy.notes)
        self.assertEqual("Plange_1979", rows[1].doc_id)

    def test_an_identical_copy_is_named_as_such(self):
        rows = [self._row("b.pdf", "title", "same"), self._row("a.pdf", "title", "same")]
        held = hold_repeated_bib_entries(rows)
        self.assertEqual("b.pdf", Path(held[0].pdf_path).name)
        self.assertIn("identical copy", held[0].notes)


if __name__ == "__main__":
    unittest.main()
