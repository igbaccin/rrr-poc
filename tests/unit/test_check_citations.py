import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[2]
CHECKER_PATH = REPO_ROOT / "scripts" / "check_citations.py"
SPEC = importlib.util.spec_from_file_location("check_citations", CHECKER_PATH)
CHECKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKER)


class CitationCheckerTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.data_dir = self.root / "data"
        self.page_text_dir = self.data_dir / "page_text"
        self.page_text_dir.mkdir(parents=True)
        self.metadata_path = self.root / "metadata.csv"
        self.docs = {
            "AcemogluEtAl_2002": 50,
            "North_1989": 20,
            "Frankema_2012": 30,
            "Frankema&vanWaijenburg_2012": 40,
            "Kuznets_1973": 20,
            "Mokyr_2005": 60,
            "Nunn_2008": 20,
            "Nunn&Wantchekon_2011": 30,
            "Skocpol&Sommers_1980": 30,
            "opaque_source": 20,
        }
        labels = {
            "opaque_source": (
                "Garcia and de Vries (2019)", "Garcia", "2019"),
        }
        authors = {
            "Skocpol&Sommers_1980": "Skocpol & Somers",
        }
        self.metadata_path.write_text(
            "doc_id,display_label,first_author_surname,year,authors\n"
            + "\n".join(
                ",".join((
                    doc_id,
                    *labels.get(doc_id, ("", "", "")),
                    authors.get(doc_id, ""),
                ))
                for doc_id in self.docs
            )
            + "\n",
            encoding="utf-8",
        )
        for doc_id, page_count in self.docs.items():
            (self.data_dir / f"{doc_id}.json").write_text(
                json.dumps({"doc_id": doc_id, "page_count": page_count}),
                encoding="utf-8",
            )
        self.write_page(
            "AcemogluEtAl_2002",
            33,
            "The outcome is described as equilibrium institutions in the text.",
        )
        self.write_page(
            "AcemogluEtAl_2002",
            5,
            "Institutions of private property are essential for investment "
            "incentives and successful economic performance.",
        )
        self.write_page(
            "North_1989",
            9,
            "This page discusses transaction costs and institutional change.",
        )
        self.write_page(
            "Frankema&vanWaijenburg_2012",
            27,
            "Real wages in British Africa stood above subsistence.",
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def write_page(self, doc_id, page, text):
        (self.page_text_dir / f"{doc_id}_page_{page}.txt").write_text(
            text,
            encoding="utf-8",
        )

    def check(self, text):
        return CHECKER.check_review(
            text,
            metadata_path=str(self.metadata_path),
            data_dir=str(self.data_dir),
        )

    def test_following_citation_in_same_sentence_governs_quote(self):
        text = (
            "North discusses transaction costs (North 1989, p.9). "
            "Acemoglu et al. call these \"equilibrium institutions\" "
            "(Acemoglu et al. 2002, p.33)."
        )
        result = self.check(text)
        self.assertEqual(result["e4"], 0)
        self.assertEqual(result["e5"], 0)
        self.assertEqual(result["quotes_verified"], 1)

    def test_canonical_citation_surface_uses_same_sentence_rule(self):
        text = (
            "North discusses transaction costs (North_1989: p.9). "
            "Acemoglu et al. call these \"equilibrium institutions\" "
            "(AcemogluEtAl_2002: p.33)."
        )
        result = self.check(text)
        self.assertEqual(result["e4"], 0)
        self.assertEqual(result["e5"], 0)
        self.assertEqual(result["quotes_verified"], 1)

    def test_any_same_sentence_citation_can_verify_quote(self):
        text = (
            "Acemoglu et al. (Acemoglu et al. 2002, p.33); "
            "(North 1989, p.9) describe \"equilibrium institutions\" "
            "as a political outcome."
        )
        result = self.check(text)
        self.assertEqual(result["e4"], 0)
        self.assertEqual(result["quotes_verified"], 1)

    def test_adjacent_sentence_citation_cannot_rescue_fabrication(self):
        text = (
            "Acemoglu et al. discuss the term (Acemoglu et al. 2002, p.33). "
            "North calls the outcome \"equilibrium institutions\" "
            "(North 1989, p.9)."
        )
        result = self.check(text)
        self.assertEqual(result["e4"], 1)
        self.assertEqual(result["quotes_verified"], 0)
        self.assertEqual(result["e4_details"][0]["candidate_scope"], "same_sentence")
        self.assertEqual(result["e4_details"][0]["doc_id"], "North_1989")

    def test_genuine_fabrication_remains_e4(self):
        text = (
            "Acemoglu et al. argue that \"the initial level of income and population density\" "
            "shaped policy (Acemoglu et al. 2002, p.33)."
        )
        result = self.check(text)
        self.assertEqual(result["e4"], 1)
        self.assertEqual(result["e5"], 0)

    def test_wrong_page_in_same_document_is_e5(self):
        text = (
            "Acemoglu et al. describe \"equilibrium institutions\" "
            "(Acemoglu et al. 2002, p.34)."
        )
        self.write_page("AcemogluEtAl_2002", 34, "A different passage.")
        result = self.check(text)
        self.assertEqual(result["e4"], 0)
        self.assertEqual(result["e5"], 1)
        self.assertEqual(result["e5_details"][0]["found_on_pages"], [33])

    def test_punctuation_inside_quote_mark_does_not_create_false_e5(self):
        self.write_page(
            "AcemogluEtAl_2002",
            34,
            "The phrase equilibrium institutions appears without a comma.",
        )
        text = (
            "Acemoglu et al. describe \"equilibrium institutions,\" "
            "before extending the argument (Acemoglu et al. 2002, p.34)."
        )
        result = self.check(text)
        self.assertEqual(result["e4"], 0)
        self.assertEqual(result["e5"], 0)
        self.assertEqual(result["quotes_verified"], 1)

    def test_pdf_ligature_and_soft_hyphen_normalize_for_quote_match(self):
        self.write_page(
            "AcemogluEtAl_2002",
            34,
            "Efﬁcient rules can produce weaker \u00adinstitutions.",
        )
        text = (
            "The authors write that \"efficient rules can produce weaker institutions\" "
            "(Acemoglu et al. 2002, p.34)."
        )
        result = self.check(text)
        self.assertEqual(result["e4"], 0)
        self.assertEqual(result["e5"], 0)
        self.assertEqual(result["quotes_verified"], 1)

    def test_unicode_dashes_normalize_symmetrically(self):
        self.write_page(
            "AcemogluEtAl_2002",
            34,
            "Winnerssecure property from 1600-1848.",
        )
        text = (
            "The authors write that \"winners—secure property from 1600–1848\" "
            "(Acemoglu et al. 2002, p.34)."
        )
        result = self.check(text)
        self.assertEqual(result["e4"], 0)
        self.assertEqual(result["e5"], 0)
        self.assertEqual(result["quotes_verified"], 1)

    def test_ocr_currency_marker_normalizes_only_before_digits(self):
        self.write_page(
            "AcemogluEtAl_2002",
            34,
            "Borrowing rose from ?1 million to ?17 million.",
        )
        text = (
            "The authors report that \"borrowing rose from Ł1 million to £17 million\" "
            "(Acemoglu et al. 2002, p.34)."
        )
        result = self.check(text)
        self.assertEqual(result["e4"], 0)
        self.assertEqual(result["e5"], 0)
        self.assertEqual(result["quotes_verified"], 1)

    def test_correct_page_in_next_sentence_cannot_hide_e5(self):
        text = (
            "Acemoglu et al. describe \"equilibrium institutions\" "
            "(Acemoglu et al. 2002, p.34). "
            "The term also appears in their discussion (Acemoglu et al. 2002, p.33)."
        )
        self.write_page("AcemogluEtAl_2002", 34, "A different passage.")
        result = self.check(text)
        self.assertEqual(result["e4"], 0)
        self.assertEqual(result["e5"], 1)
        self.assertEqual(result["quotes_verified"], 0)
        self.assertEqual(result["e5_details"][0]["candidate_scope"], "same_sentence")

    def test_adjacent_sentence_openings_do_not_rescue_fabrication(self):
        next_sentences = (
            "(Acemoglu et al. 2002, p.33) supplies a separate discussion.",
            "“Separately,” Acemoglu et al. use the term "
            "(Acemoglu et al. 2002, p.33).",
            "a separate sentence cites the term "
            "(Acemoglu et al. 2002, p.33).",
        )
        for next_sentence in next_sentences:
            with self.subTest(next_sentence=next_sentence):
                text = (
                    "North calls the outcome \"equilibrium institutions\" "
                    "(North 1989, p.9). " + next_sentence
                )
                result = self.check(text)
                self.assertEqual(result["e4"], 1)
                self.assertEqual(result["quotes_verified"], 0)
                self.assertEqual(
                    result["e4_details"][0]["candidate_cites"],
                    ["North_1989 p.9"],
                )

    def test_same_sentence_citation_is_not_limited_to_fallback_window(self):
        text = (
            "North gives another account (North 1989, p.9). "
            "Acemoglu et al. call these \"equilibrium institutions\" "
            + "within a deliberately extended discussion " * 8
            + "(Acemoglu et al. 2002, p.33)."
        )
        self.assertGreater(
            text.index("(Acemoglu et al. 2002")
            - text.index("equilibrium institutions"),
            CHECKER._QUOTE_WINDOW_CHARS,
        )
        result = self.check(text)
        self.assertEqual(result["e4"], 0)
        self.assertEqual(result["e5"], 0)
        self.assertEqual(result["quotes_verified"], 1)

    def test_author_abbreviation_does_not_split_same_sentence(self):
        text = (
            "Acemoglu et al. (Acemoglu et al. 2002, p.33) call these "
            "\"equilibrium institutions\" in their account."
        )
        quote_start = text.index('"equilibrium institutions"')
        sentence_start, _ = CHECKER._sentence_bounds(
            text, quote_start, quote_start + len('"equilibrium institutions"'))
        self.assertEqual(sentence_start, 0)
        result = self.check(text)
        self.assertEqual(result["e4"], 0)
        self.assertEqual(result["quotes_verified"], 1)

    def test_internal_ellipsis_does_not_end_sentence(self):
        text = (
            "North gives another account (North 1989, p.9). "
            "Acemoglu et al. call these \"equilibrium institutions\" ... "
            + "within a deliberately extended discussion " * 8
            + "(Acemoglu et al. 2002, p.33)."
        )
        result = self.check(text)
        self.assertEqual(result["e4"], 0)
        self.assertEqual(result["e5"], 0)
        self.assertEqual(result["quotes_verified"], 1)

    def test_ellipsis_fragments_match_omitted_source_words_in_order(self):
        self.write_page(
            "AcemogluEtAl_2002",
            34,
            "A cluster of institutions of private property are essential "
            "for long-run economic performance.",
        )
        text = (
            "The authors describe \"a cluster ... are essential ... performance\" "
            "(Acemoglu et al. 2002, p.34)."
        )
        result = self.check(text)
        self.assertEqual(result["e4"], 0)
        self.assertEqual(result["e5"], 0)
        self.assertEqual(result["quotes_verified"], 1)

    def test_ellipsis_fragments_in_wrong_order_remain_unverified(self):
        text = (
            "The authors describe \"performance ... a cluster of institutions\" "
            "(Acemoglu et al. 2002, p.33)."
        )
        result = self.check(text)
        self.assertEqual(result["e4"], 1)
        self.assertEqual(result["quotes_verified"], 0)

    def test_editorial_brackets_match_surrounding_source_fragments(self):
        self.write_page(
            "AcemogluEtAl_2002",
            34,
            "The government replicated European institutions and became "
            "financially solvent.",
        )
        text = (
            "The account says \"the government replicate[d] European institutions "
            "and became [fully] financially solvent\" "
            "(Acemoglu et al. 2002, p.34)."
        )
        result = self.check(text)
        self.assertEqual(result["e4"], 0)
        self.assertEqual(result["e5"], 0)
        self.assertEqual(result["quotes_verified"], 1)

    def test_abbreviation_at_sentence_end_does_not_merge_next_citation(self):
        text = (
            "North calls the outcome \"equilibrium institutions\" "
            "(North 1989, p.9) in a discussion of the U.S. "
            "Acemoglu et al. supply a separate discussion "
            "(Acemoglu et al. 2002, p.33)."
        )
        result = self.check(text)
        self.assertEqual(result["e4"], 1)
        self.assertEqual(result["quotes_verified"], 0)
        self.assertEqual(
            result["e4_details"][0]["candidate_cites"],
            ["North_1989 p.9"],
        )

    def test_full_coauthor_signature_resolves_particle_variant(self):
        text = (
            "The wage evidence is reported in "
            "(Frankema and Waijenburg 2012, p.27)."
        )
        result = self.check(text)
        self.assertEqual(result["e1"], 0)
        self.assertEqual(result["e2"], 0)
        self.assertIn("Frankema&vanWaijenburg_2012", result["docs_cited"])
        self.assertNotIn("Frankema_2012", result["docs_cited"])

    def test_mismatched_multi_author_label_does_not_degrade_to_coauthor(self):
        result = self.check(
            "The claim appears in Mokyr and Kuznets (1973, p.1).")
        self.assertEqual(result["e1"], 1)
        self.assertNotIn("Kuznets_1973", result["docs_cited"])

    def test_later_coauthor_alone_does_not_resolve(self):
        result = self.check("The claim appears in (Wantchekon 2011, p.1).")
        self.assertEqual(result["e1"], 1)
        self.assertNotIn("Nunn&Wantchekon_2011", result["docs_cited"])

    def test_full_doc_id_coauthor_signature_still_resolves(self):
        result = self.check(
            "The claim appears in (Nunn and Wantchekon 2011, p.1).")
        self.assertEqual(result["e1"], 0)
        self.assertIn("Nunn&Wantchekon_2011", result["docs_cited"])

    def test_unique_first_author_et_al_still_resolves(self):
        result = self.check("The claim appears in (Nunn et al. 2011, p.1).")
        self.assertEqual(result["e1"], 0)
        self.assertIn("Nunn&Wantchekon_2011", result["docs_cited"])

    def test_metadata_coauthor_signature_resolves_particle_variant(self):
        result = self.check(
            "The claim appears in (Garcia and Vries 2019, p.1).")
        self.assertEqual(result["e1"], 0)
        self.assertIn("opaque_source", result["docs_cited"])

    def test_metadata_signature_overrides_misspelled_legacy_doc_id(self):
        result = self.check(
            "The claim appears in (Skocpol and Somers 1980, p.24).")
        self.assertEqual(result["e1"], 0)
        self.assertIn("Skocpol&Sommers_1980", result["docs_cited"])

    def test_full_display_group_counts_each_unit_once(self):
        result = self.check(
            "The comparison is established "
            "(Nunn and Wantchekon 2011, p.27; Nunn 2008, p.2).")
        self.assertEqual(result["n_citations"], 2)
        self.assertEqual(result["n_canonical_citations"], 0)
        self.assertEqual(result["n_display_citations"], 2)
        self.assertEqual(result["e1"], 0)
        self.assertEqual(result["e2"], 0)
        self.assertEqual(result["e3"], 0)
        self.assertEqual(
            result["docs_cited"],
            ["Nunn&Wantchekon_2011", "Nunn_2008"],
        )

    def test_display_group_can_wrap_across_lines(self):
        result = self.check(
            "The comparison is established "
            "(Nunn and Wantchekon 2011, p.27;\nNunn 2008, p.2).")
        self.assertEqual(result["n_citations"], 2)
        self.assertEqual(result["e1"], 0)
        self.assertEqual(result["e2"], 0)
        self.assertEqual(result["e3"], 0)
        self.assertEqual(
            result["docs_cited"],
            ["Nunn&Wantchekon_2011", "Nunn_2008"],
        )

    def test_full_canonical_group_does_not_double_count_members(self):
        result = self.check(
            "The comparison is established "
            "(Nunn&Wantchekon_2011: p.27; North_1989: p.9).")
        self.assertEqual(result["n_citations"], 2)
        self.assertEqual(result["n_canonical_citations"], 2)
        self.assertEqual(result["n_display_citations"], 0)
        self.assertEqual(result["e1"], 0)
        self.assertEqual(result["e2"], 0)
        self.assertEqual(result["e3"], 0)

    def test_narrative_prefix_group_parses_first_unit_with_its_author(self):
        result = self.check(
            "Nunn and Wantchekon (2011,p.27; Nunn 2008,p.2) "
            "provide the comparison.")
        self.assertEqual(result["n_citations"], 2)
        self.assertEqual(result["n_display_citations"], 2)
        self.assertEqual(result["e1"], 0)
        self.assertEqual(result["e3"], 0)
        self.assertEqual(
            result["docs_cited"],
            ["Nunn&Wantchekon_2011", "Nunn_2008"],
        )

    def test_group_members_receive_independent_e1_and_e2_checks(self):
        result = self.check(
            "The comparison is established "
            "(Nunn 2008,p.2; Imaginary 2099,p.1; North 1989,p.99).")
        self.assertEqual(result["n_citations"], 3)
        self.assertEqual(result["e1"], 1)
        self.assertEqual(result["e2"], 1)
        self.assertEqual(result["e3"], 0)
        self.assertEqual(result["e1_details"][0]["label"], "Imaginary")
        self.assertEqual(result["e2_details"][0]["doc_id"], "North_1989")

        canonical = self.check(
            "The comparison is established "
            "(Nunn_2008:p.2; Imaginary_2099:p.1; North_1989:p.99).")
        self.assertEqual(canonical["n_citations"], 3)
        self.assertEqual(canonical["n_canonical_citations"], 3)
        self.assertEqual(canonical["e1"], 1)
        self.assertEqual(canonical["e2"], 1)
        self.assertEqual(canonical["e3"], 0)

    def test_malformed_group_member_is_one_hard_e3_and_never_inherits(self):
        malformed = {
            "page_only": "(Nunn 2008,p.2; p.3)",
            "page_range": "(Nunn 2008,p.2; Nunn 2008,p.2-3)",
            "comma_multi_page": "(Nunn 2008,p.2; Nunn 2008,p.2,p.3)",
            "author_year_only": "(Nunn 2008,p.2; North 1989)",
        }
        for label, citation in malformed.items():
            with self.subTest(label=label):
                result = self.check("The comparison is established " + citation + ".")
                self.assertEqual(result["n_citations"], 1)
                self.assertEqual(result["n_display_citations"], 1)
                self.assertEqual(result["e1"], 0)
                self.assertEqual(result["e2"], 0)
                self.assertEqual(result["e3"], 1)
                self.assertEqual(result["e3_soft"], 0)
                self.assertEqual(result["docs_cited"], ["Nunn_2008"])

    def test_malformed_canonical_group_member_is_not_inherited(self):
        malformed = {
            "page_only": "(North_1989:p.9; p.10)",
            "page_range": "(North_1989:p.9; North_1989:p.10-p.11)",
            "comma_multi_page": "(North_1989:p.9; North_1989:p.10,p.11)",
            "author_year_only": "(North_1989:p.9; Nunn 2008)",
        }
        for label, citation in malformed.items():
            with self.subTest(label=label):
                result = self.check("The comparison is established " + citation + ".")
                self.assertEqual(result["n_citations"], 1)
                self.assertEqual(result["n_canonical_citations"], 1)
                self.assertEqual(result["e1"], 0)
                self.assertEqual(result["e2"], 0)
                self.assertEqual(result["e3"], 1)
                self.assertEqual(result["docs_cited"], ["North_1989"])

    def test_semicolon_prose_with_year_is_not_a_group_format_error(self):
        result = self.check(
            "The comparison is contextual (in 1900; Nunn 2008,p.2).")
        self.assertEqual(result["e3"], 0)
        self.assertEqual(result["n_citations"], 0)

    def test_repeated_group_units_count_twice_and_document_once(self):
        result = self.check(
            "The comparison is established "
            "(Nunn 2008,p.2; Nunn 2008,p.2).")
        self.assertEqual(result["n_citations"], 2)
        self.assertEqual(result["n_docs_cited"], 1)
        self.assertEqual(result["docs_cited"], ["Nunn_2008"])

    def test_quote_can_verify_against_any_same_sentence_group_member(self):
        text = (
            "The authors call these \"equilibrium institutions\" "
            "(North 1989,p.9; Acemoglu et al. 2002,p.33)."
        )
        result = self.check(text)
        self.assertEqual(result["quotes_checked"], 1)
        self.assertEqual(result["quotes_verified"], 1)
        self.assertEqual(result["e4"], 0)
        self.assertEqual(result["e5"], 0)

    def test_nearest_fallback_keeps_every_member_of_selected_group(self):
        text = (
            "The phrase \"equilibrium institutions\" appears in the account. "
            "Sources follow (North 1989,p.9; Acemoglu et al. 2002,p.33)."
        )
        result = self.check(text)
        self.assertEqual(result["quotes_checked"], 1)
        self.assertEqual(result["quotes_verified"], 1)
        self.assertEqual(result["e4"], 0)
        self.assertEqual(result["e5"], 0)

    def test_grouped_container_inside_quote_is_excluded_from_quote_scan(self):
        text = (
            "The prose says \"a quoted span (Nunn 2008,p.2; "
            "North 1989,p.9) embedded in text\" (North 1989,p.9)."
        )
        result = self.check(text)
        self.assertEqual(result["n_citations"], 3)
        self.assertEqual(result["quotes_checked"], 0)
        self.assertEqual(result["e4"], 0)
        self.assertEqual(result["e5"], 0)

    def test_valid_inner_citation_does_not_hide_malformed_outer_wrapper(self):
        result = self.check(
            "The claim appears in "
            "(Mokyr_2005 (Nunn 2008, p.2))."
        )
        self.assertEqual(result["n_citations"], 1)
        self.assertEqual(result["e3"], 1)
        self.assertEqual(result["e3_details"][0]["reason"], "doc_without_page")

    def test_malformed_range_is_quote_only_evidence_and_remains_e3(self):
        result = self.check(
            "The authors say that institutions \"are essential for investment "
            "incentives and successful economic performance\" "
            "(AcemogluEtAl_2002: p.5-6)."
        )
        self.assertEqual(result["n_citations"], 0)
        self.assertEqual(result["e3"], 1)
        self.assertEqual(result["quotes_checked"], 1)
        self.assertEqual(result["quotes_verified"], 1)
        self.assertEqual(result["e4"], 0)
        self.assertEqual(result["e5"], 0)

    def test_display_range_is_quote_only_evidence_and_remains_e3(self):
        result = self.check(
            "The authors say that institutions \"are essential for investment "
            "incentives and successful economic performance\" "
            "(Acemoglu et al. 2002, p.5-6)."
        )
        self.assertEqual(result["n_citations"], 0)
        self.assertEqual(result["e3"], 1)
        self.assertEqual(result["quotes_checked"], 1)
        self.assertEqual(result["quotes_verified"], 1)
        self.assertEqual(result["e4"], 0)
        self.assertEqual(result["e5"], 0)

    def test_page_only_group_member_is_quote_only_evidence_and_remains_e3(self):
        self.write_page(
            "Nunn_2008",
            3,
            "The slave trades resulted in the evolution of institutions that "
            "were not conducive to economic growth.",
        )
        result = self.check(
            "The trade \"resulted in the evolution of institutions that were "
            "not conducive to economic growth\" (Nunn_2008: p.1; p.3)."
        )
        self.assertEqual(result["n_citations"], 1)
        self.assertEqual(result["e3"], 1)
        self.assertEqual(result["quotes_checked"], 1)
        self.assertEqual(result["quotes_verified"], 1)
        self.assertEqual(result["e4"], 0)
        self.assertEqual(result["e5"], 0)

    def test_fallback_keeps_all_citations_in_nearest_citation_sentence(self):
        result = self.check(
            "The prior account cites both sources (Acemoglu et al. 2002, p.33) "
            "and (North 1989, p.9). The next sentence repeats the phrase "
            "\"equilibrium institutions\" without another citation."
        )
        self.assertEqual(result["quotes_checked"], 1)
        self.assertEqual(result["quotes_verified"], 1)
        self.assertEqual(result["e4"], 0)
        self.assertEqual(result["e5"], 0)


if __name__ == "__main__":
    unittest.main()
