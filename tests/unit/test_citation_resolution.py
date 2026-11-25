import unittest

import rrr.render as render
import rrr.writer as writer


def _labels(rows):
    render.set_metadata_labels(rows)


def _evidence(*triples):
    return {eid: {"doc_id": doc_id, "page": page} for eid, doc_id, page in triples}


def _allowed(evidence):
    docs = {item["doc_id"] for item in evidence.values()}
    pairs = {(item["doc_id"], item["page"]) for item in evidence.values()}
    return docs, pairs


class _RenderCase(unittest.TestCase):
    def tearDown(self):
        render.set_metadata_labels([])


class YearLetterTests(_RenderCase):
    def test_catalogue_year_letter_reaches_the_label_and_the_citation(self):
        _labels([
            {"doc_id": "Plange_1979a", "authors": "Plange", "year": "1979a"},
            {"doc_id": "Plange_1979b", "authors": "Plange", "year": "1979b"},
        ])
        self.assertEqual("Plange (1979a)", render._doc_id_to_author_label("Plange_1979a"))
        self.assertEqual("Plange (1979b)", render._doc_id_to_author_label("Plange_1979b"))
        self.assertEqual("Plange (1979a, p.19)", render.render_citation("Plange_1979a", 19))

        evidence = _evidence(("E0001", "Plange_1979a", 19), ("E0002", "Plange_1979b", 3))
        text, stats = writer._render_evidence_id_citations(
            "Labour was recruited by force [E0001]. Plange [E0002] dates the reserve.",
            evidence,
        )
        self.assertEqual(2, stats["replacements"])
        self.assertIn("(Plange 1979a, p.19)", text)
        self.assertIn("Plange (1979b, p.3)", text)

        docs, pairs = _allowed(evidence)
        lookup = render._build_display_lookup(docs)
        parsed = list(render.parse_citations(text, display_lookup=lookup))
        self.assertEqual(
            [("Plange_1979a", 19, "1979a"), ("Plange_1979b", 3, "1979b")],
            [(c["doc_id"], c["page"], c["year"]) for c in parsed],
        )
        cleaned, removed = writer._remove_invalid_citations(
            text, docs, pairs, display_lookup=lookup,
        )
        self.assertEqual([], removed)
        self.assertEqual(text, cleaned)

    def test_the_pilot_catalogue_no_longer_fails_the_release_check(self):
        _labels([
            {"doc_id": "Plange_1979", "authors": "Plange", "year": 1979},
            {"doc_id": "Plange_1979a", "authors": "Plange", "year": 1979},
            {"doc_id": "Brukum_1998", "authors": "Brukum", "year": 1998},
        ])
        self.assertEqual("Plange (1979)", render._doc_id_to_author_label("Plange_1979"))
        self.assertEqual("Plange (1979a)", render._doc_id_to_author_label("Plange_1979a"))

        evidence = _evidence(
            ("E0001", "Plange_1979", 19),
            ("E0002", "Plange_1979a", 1),
            ("E0003", "Plange_1979a", 3),
            ("E0004", "Brukum_1998", 5),
        )
        draft = (
            "British rule turned the north into a labour reserve [E0001] [E0002]. "
            "The administration treated labour as a contribution owed [E0003]. "
            "Brukum [E0004] reads this as studied neglect."
        )
        text, _ = writer._render_evidence_id_citations(draft, evidence)
        docs, pairs = _allowed(evidence)
        lookup = render._build_display_lookup(docs)
        cleaned, removed = writer._remove_invalid_citations(
            text, docs, pairs, display_lookup=lookup,
        )
        self.assertEqual([], removed)
        self.assertEqual(
            ["Plange_1979", "Plange_1979a", "Plange_1979a", "Brukum_1998"],
            [c["doc_id"] for c in render.parse_citations(cleaned, display_lookup=lookup)],
        )

    def test_a_catalogue_without_a_pair_keeps_its_labels(self):
        _labels([
            {"doc_id": "Acemoglu_2001a", "authors": "Acemoglu", "year": 2001},
            {"doc_id": "North_1989", "authors": "North", "year": 1989},
        ])
        self.assertEqual("Acemoglu (2001)", render._doc_id_to_author_label("Acemoglu_2001a"))
        self.assertEqual("North (1989)", render._doc_id_to_author_label("North_1989"))
        self.assertEqual("North (1989, p.4)", render.render_citation("North_1989", 4))

    def test_a_letter_the_catalogue_does_not_use_falls_back_to_the_plain_year(self):
        _labels([{"doc_id": "North_1989", "authors": "North", "year": 1989}])
        lookup = render._build_display_lookup({"North_1989"})
        parsed = list(render.parse_citations("North (1989a, p.3) agrees.", display_lookup=lookup))
        self.assertEqual(["North_1989"], [c["doc_id"] for c in parsed])
        self.assertEqual("North_1989", lookup.get(("north", "1989a")))

    def test_a_wrong_letter_stays_unresolved(self):
        _labels([
            {"doc_id": "Plange_1979a", "authors": "Plange", "year": "1979a"},
            {"doc_id": "Plange_1979b", "authors": "Plange", "year": "1979b"},
        ])
        lookup = render._build_display_lookup({"Plange_1979a", "Plange_1979b"})
        parsed = list(render.parse_citations("Plange (1979c, p.3) agrees.", display_lookup=lookup))
        self.assertEqual([None], [c["doc_id"] for c in parsed])

    def test_a_single_lettered_work_answers_to_the_plain_year_too(self):
        _labels([{"doc_id": "Plange_1979a", "authors": "Plange", "year": "1979a"}])
        lookup = render._build_display_lookup({"Plange_1979a"})
        for text in ("Plange (1979a, p.3) agrees.", "Plange (1979, p.3) agrees."):
            parsed = list(render.parse_citations(text, display_lookup=lookup))
            self.assertEqual(["Plange_1979a"], [c["doc_id"] for c in parsed], text)

    def test_year_cells_from_pandas_are_read(self):
        self.assertEqual("1979", render._year_cell(1979))
        self.assertEqual("1979", render._year_cell(1979.0))
        self.assertEqual("1979a", render._year_cell(" 1979a "))
        self.assertEqual("", render._year_cell(float("nan")))
        self.assertEqual("", render._year_cell(None))

    def test_pageless_author_year_lookup_knows_the_letter(self):
        _labels([
            {"doc_id": "Plange_1979a", "first_author_surname": "Plange", "year": "1979a"},
            {"doc_id": "Plange_1979b", "first_author_surname": "Plange", "year": "1979b"},
        ])
        docs = ["Plange_1979a", "Plange_1979b"]
        lookup = render._build_author_year_lookup(docs)
        self.assertEqual("Plange_1979a", lookup[("plange", "1979a")])
        self.assertEqual("Plange_1979b", lookup[("plange", "1979b")])
        cited = render._collect_cited_docs("As (Plange, 1979a) notes.", set(docs), lookup)
        self.assertEqual({"Plange_1979a"}, cited)


class SharedLabelTests(_RenderCase):
    ROWS = [
        {"doc_id": "ReportNorth_2000", "display_label": "Service (2000)", "year": 2000},
        {"doc_id": "ReportSouth_2000", "display_label": "Service (2000)", "year": 2000},
    ]

    def _render(self):
        _labels(self.ROWS)
        evidence = _evidence(
            ("E0001", "ReportNorth_2000", 19), ("E0002", "ReportSouth_2000", 3),
        )
        text, _ = writer._render_evidence_id_citations(
            "Poverty fell in the south [E0002]. It rose in the north [E0001].",
            evidence,
        )
        docs, pairs = _allowed(evidence)
        return text, docs, pairs, render._build_display_lookup(docs)

    def test_the_shared_key_is_listed_as_ambiguous(self):
        _, _, _, lookup = self._render()
        self.assertNotIn(("service", "2000"), lookup)
        self.assertEqual(
            ("ReportNorth_2000", "ReportSouth_2000"),
            lookup.ambiguous[("service", "2000")],
        )
        self.assertIsNone(lookup.get(("service", "2000")))

    def test_the_citation_is_read_back_as_the_document_it_was_rendered_for(self):
        text, docs, pairs, lookup = self._render()
        parsed = list(render.parse_citations(text, display_lookup=lookup))
        self.assertEqual(
            [("ReportSouth_2000", 3), ("ReportNorth_2000", 19)],
            [(c["doc_id"], c["page"]) for c in parsed],
        )
        self.assertTrue(all(c["candidates"] for c in parsed))
        cleaned, removed = writer._remove_invalid_citations(
            text, docs, pairs, display_lookup=lookup,
        )
        self.assertEqual([], removed)
        self.assertEqual(text, cleaned)

    def test_the_release_check_holds_without_the_render_record(self):
        text, docs, pairs, lookup = self._render()
        render._RENDERED_CITATIONS.clear()
        cleaned, removed = writer._remove_invalid_citations(
            text, docs, pairs, display_lookup=lookup,
        )
        self.assertEqual([], removed)
        for citation in render.parse_citations(cleaned, display_lookup=lookup):
            self.assertIsNotNone(citation["doc_id"])
            self.assertIsNotNone(render.citation_inside(citation, docs, pairs))

    def test_a_page_no_candidate_allows_is_still_rejected(self):
        text, docs, pairs, lookup = self._render()
        forged = text + " A further claim (Service 2000, p.77)."
        _, removed = writer._remove_invalid_citations(
            forged, docs, pairs, display_lookup=lookup,
        )
        self.assertEqual(1, len(removed))
        self.assertEqual("invalid_page", removed[0]["reason"])
        self.assertEqual(77, removed[0]["page"])


class NameFormTests(_RenderCase):
    DOC_IDS = [
        "deHaas_2017",
        "vanderGeest_2011",
        "Frankema&vanWaijenburg_2012",
        "delaCroix&Doepke_2003",
        "MeierzuSelhausen_2018",
        "elSayed_2001",
        "AryeeteyAttoh&Chatterjee_1988",
        "GhanaStatisticalService_2000a",
        "GhanaStatisticalService_2000b",
        "WorldBank_1995",
        "Canagarajah&Portner_2003",
        "AppiahKubiEtAl_2008",
        "vanZanden_2009",
        "laPorta_1998",
        "tenBrink_1990",
    ]

    def test_labels_split_the_particles_back_out(self):
        expected = {
            "deHaas_2017": "de Haas (2017)",
            "vanderGeest_2011": "van der Geest (2011)",
            "Frankema&vanWaijenburg_2012": "Frankema and van Waijenburg (2012)",
            "delaCroix&Doepke_2003": "de la Croix and Doepke (2003)",
            "vanZanden_2009": "van Zanden (2009)",
            "elSayed_2001": "ElSayed (2001)",
            "MeierzuSelhausen_2018": "MeierzuSelhausen (2018)",
        }
        for doc_id, label in expected.items():
            self.assertEqual(label, render._regex_doc_id_to_author_label(doc_id), doc_id)

    def test_every_label_round_trips_in_both_citation_forms(self):
        rows = [
            {"doc_id": doc_id, "year": doc_id.rsplit("_", 1)[1]}
            for doc_id in self.DOC_IDS
        ]
        _labels(rows)
        lookup = render._build_display_lookup(self.DOC_IDS)
        self.assertEqual(
            {("ghanastatisticalservice", "2000"): (
                "GhanaStatisticalService_2000a", "GhanaStatisticalService_2000b",
            )},
            lookup.ambiguous,
        )
        for doc_id in self.DOC_IDS:
            narrative = render.render_citation(doc_id, 3)
            label, _, tail = narrative.rpartition(" (")
            parenthetical = f"({label} {tail}"
            for text in (
                f"As {narrative} shows, this holds.",
                f"This holds {parenthetical}.",
                f"This holds (Austin 2008, p.1; {parenthetical[1:]}.",
            ):
                found = [
                    c["doc_id"]
                    for c in render.parse_citations(text, display_lookup=lookup)
                    if c["label"] != "Austin"
                ]
                self.assertEqual([doc_id], found, text)

    def test_the_renderer_and_the_release_check_agree_on_particle_names(self):
        _labels([
            {"doc_id": "vanderGeest_2011", "year": 2011},
            {"doc_id": "deHaas_2017", "year": 2017},
        ])
        evidence = _evidence(("E0001", "vanderGeest_2011", 4), ("E0002", "deHaas_2017", 9))
        text, _ = writer._render_evidence_id_citations(
            "Migration was seasonal [E0001], and de Haas [E0002] measures wages.",
            evidence,
        )
        self.assertIn("(van der Geest 2011, p.4)", text)
        self.assertIn("de Haas (2017, p.9)", text)
        docs, pairs = _allowed(evidence)
        lookup = render._build_display_lookup(docs)
        _, removed = writer._remove_invalid_citations(
            text, docs, pairs, display_lookup=lookup,
        )
        self.assertEqual([], removed)

    def test_a_particle_word_before_a_label_does_not_hide_the_document(self):
        _labels([{"doc_id": "North_1990", "year": 1990}])
        lookup = render._build_display_lookup({"North_1990"})
        parsed = list(render.parse_citations(
            "Of the ten North (1990, p.3) lists, two matter.", display_lookup=lookup,
        ))
        self.assertEqual(["North_1990"], [c["doc_id"] for c in parsed])


if __name__ == "__main__":
    unittest.main()
