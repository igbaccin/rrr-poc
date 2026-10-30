"""A citation must never be folded into a prose phrase naming two authors.

Run 043 of the July 2026 battery released ``Mokyr and Kuznets (1973, p.1)``
after the model wrote ``Mokyr and Kuznets (1973) [E0081] [E0089]``. The
checker read the result as a two-author work outside the corpus. These tests
run the writer's production sequence (renderer, chunk normalization, final
normalization, adjacent-citation merge, repair, terminal audit) on that input
and on the legitimate forms the guards must leave alone.
"""
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rrr import writer  # noqa: E402
from rrr.render import _build_display_lookup, parse_citations  # noqa: E402

EVIDENCE = {
    "E0081": {"doc_id": "Kuznets_1973", "page": 1},
    "E0089": {"doc_id": "Mokyr_2005", "page": 49},
    "E0001": {"doc_id": "North&Weingast_1989", "page": 2},
    "E0007": {"doc_id": "North_1989", "page": 6},
}
LOOKUP = _build_display_lookup({entry["doc_id"] for entry in EVIDENCE.values()})
LEAD = "Earlier work stressed capital accumulation. "


def released(text):
    """Apply the writer's citation passes in production order."""
    text, stats = writer._render_evidence_id_citations(text, EVIDENCE)
    text, _ = writer._collapse_double_parens(text)
    text, _ = writer._collapse_double_parens(text, LOOKUP)
    text, _ = writer._merge_adjacent_paren_cites(text, display_lookup=LOOKUP)
    text, _ = writer._repair_unresolved_display_citations(text, LOOKUP)
    faults = writer._find_citation_surface_faults(text, LOOKUP)
    return text.replace(LEAD, ""), stats, faults


def resolved_docs(text):
    return [c.get("doc_id") for c in parse_citations(text, display_lookup=LOOKUP)]


class ConjoinedAuthorCitationTests(unittest.TestCase):
    def assert_clean(self, text):
        docs = resolved_docs(text)
        self.assertTrue(docs, text)
        self.assertNotIn(None, docs, text)

    def test_run_043_sentence_keeps_both_authors_and_both_citations(self):
        text, _stats, faults = released(
            LEAD + "Mokyr and Kuznets (1973) [E0081] [E0089] argue that "
            "technology drives growth."
        )
        self.assertEqual(
            "Kuznets (1973, p.1) and Mokyr (2005, p.49) argue that technology "
            "drives growth.",
            text,
        )
        self.assertEqual(["Kuznets_1973", "Mokyr_2005"], resolved_docs(text))
        self.assertEqual([], faults)

    def test_conjoined_subject_without_bare_year_is_not_fused(self):
        text, stats, faults = released(
            LEAD + "Mokyr and Kuznets [E0081] [E0089] argue the point."
        )
        self.assertEqual(
            "Kuznets (1973, p.1) and Mokyr (2005, p.49) argue the point.", text,
        )
        self.assertEqual(1, stats["conjoined_subject_parentheticals"])
        self.assertEqual([], faults)

    def test_unknown_conjoined_name_keeps_the_parenthetical(self):
        text, _stats, faults = released(
            LEAD + "Smith and Kuznets (1973) [E0081] argue the point."
        )
        self.assertEqual("Smith and Kuznets (Kuznets 1973, p.1) argue the point.", text)
        self.assert_clean(text)
        self.assertEqual([], faults)

    def test_legitimate_forms_are_unchanged(self):
        cases = {
            LEAD + "North and Weingast (1989) [E0001] show that commitment mattered.":
                "North and Weingast (1989, p.2) show that commitment mattered.",
            LEAD + "Kuznets (1973) [E0081] notes the pattern.":
                "Kuznets (1973, p.1) notes the pattern.",
            LEAD + "Kuznets [E0081] notes the pattern.":
                "Kuznets (1973, p.1) notes the pattern.",
            LEAD + "Kuznets's [E0081] view differs.":
                "Kuznets's (1973, p.1) view differs.",
            LEAD + "Both Mokyr [E0089] and Kuznets [E0081] argue the point.":
                "Both Mokyr (2005, p.49) and Kuznets (1973, p.1) argue the point.",
            LEAD + "North (1989) [E0007] and North and Weingast (1989) [E0001] differ.":
                "North (1989, p.6) and North and Weingast (1989, p.2) differ.",
        }
        for source, expected in cases.items():
            with self.subTest(source=source):
                text, _stats, faults = released(source)
                self.assertEqual(expected, text)
                self.assertEqual([], faults)
                if "'s" not in expected:
                    # A possessive narrative form is prose to the parser.
                    self.assert_clean(text)

    def test_every_release_is_stable_on_a_second_pass(self):
        for source in (
            LEAD + "Mokyr and Kuznets (1973) [E0081] [E0089] argue the point.",
            LEAD + "Both Mokyr [E0089] and Kuznets [E0081] argue the point.",
        ):
            first, *_ = released(source)
            second, *_ = released(LEAD + first)
            self.assertEqual(first, second)

    def test_already_fused_label_is_rebound_before_the_audit(self):
        text = "Mokyr and Kuznets (1973, p.1) (Mokyr 2005, p.49) argue the point."
        repaired, count = writer._repair_unresolved_display_citations(text, LOOKUP)
        self.assertEqual(1, count)
        self.assertEqual(
            "Mokyr and Kuznets (Kuznets 1973, p.1) (Mokyr 2005, p.49) argue the point.",
            repaired,
        )
        self.assert_clean(repaired)

    def test_unrepairable_display_citation_blocks_release(self):
        text = "Smith and Jones (1973, p.1) argue the point."
        repaired, count = writer._repair_unresolved_display_citations(text, LOOKUP)
        self.assertEqual(0, count)
        faults = writer._find_citation_surface_faults(repaired, LOOKUP)
        self.assertIn("unresolved_display_citation", [f["type"] for f in faults])
        self.assertIn(
            "unresolved_display_citation", writer._BLOCKING_CITATION_SURFACE_FAULTS,
        )

    def test_page_free_mentions_stay_cosmetic(self):
        faults = writer._find_citation_surface_faults(
            "Kuznets (1973) argues the point, as the data show (Kuznets 1973, p.1).",
            LOOKUP,
        )
        types = [f["type"] for f in faults]
        self.assertIn("page_free_author_year", types)
        self.assertNotIn("unresolved_display_citation", types)


if __name__ == "__main__":
    unittest.main()
