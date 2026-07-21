import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rrr.render import parse_citations
from rrr.writer import _emit_citations_manifest


DISPLAY_LOOKUP = {
    ("austin", "2008"): "Austin_2008",
    ("north", "1989"): "North_1989",
    ("north and weingast", "1989"): "North&Weingast_1989",
    ("north & weingast", "1989"): "North&Weingast_1989",
    ("o'brien and van zanden", "2009"): "OBrien&vanZanden_2009",
    ("o\u2019brien and van zanden", "2009"): "OBrien&vanZanden_2009",
}


class GroupedCitationParserTests(unittest.TestCase):
    def test_display_group_is_memberwise_and_source_ordered(self):
        text = (
            "North (1989, p.9) opens. "
            "Then (Austin 2008, p.1; North and Weingast 1989, p.2). "
            "Finally (North_1989: p.3)."
        )

        citations = list(parse_citations(text, display_lookup=DISPLAY_LOOKUP))

        self.assertEqual(
            [citation["doc_id"] for citation in citations],
            ["North_1989", "Austin_2008", "North&Weingast_1989", "North_1989"],
        )
        self.assertEqual(
            [citation["raw"] for citation in citations],
            [
                "North (1989, p.9)",
                "Austin 2008, p.1",
                "North and Weingast 1989, p.2",
                "(North_1989: p.3)",
            ],
        )
        self.assertEqual(
            [citation["surface"] for citation in citations],
            ["display_narrative", "display_paren", "display_paren", "canonical"],
        )
        for citation in citations:
            self.assertEqual(
                text[citation["start"]:citation["end"]], citation["raw"]
            )
        for previous, current in zip(citations, citations[1:]):
            self.assertLessEqual(previous["end"], current["start"])

    def test_canonical_group_yields_each_member_once(self):
        text = "Claim (Austin_2008: p.1; North_1989: p.2)."

        citations = list(parse_citations(text))

        self.assertEqual(
            [(c["doc_id"], c["page"], c["raw"]) for c in citations],
            [
                ("Austin_2008", 1, "Austin_2008: p.1"),
                ("North_1989", 2, "North_1989: p.2"),
            ],
        )

    def test_narrative_prefix_can_supply_first_group_label(self):
        text = "Austin (2008, p.1; North 1989, p.2) argues this."

        citations = list(parse_citations(text, display_lookup=DISPLAY_LOOKUP))

        self.assertEqual(
            [(c["doc_id"], c["surface"], c["raw"]) for c in citations],
            [
                ("Austin_2008", "display_narrative", "2008, p.1"),
                ("North_1989", "display_paren", "North 1989, p.2"),
            ],
        )
        for citation in citations:
            self.assertEqual(
                text[citation["start"]:citation["end"]], citation["raw"]
            )

    def test_partly_malformed_semicolon_parenthetical_is_rejected(self):
        text = "Claim (Austin 2008, p.1; unsupported prose)."

        self.assertEqual(
            list(parse_citations(text, display_lookup=DISPLAY_LOOKUP)), []
        )

    def test_unknown_display_member_remains_unresolved(self):
        text = "Claim (Austin 2008, p.1; Unknown 2009, p.2)."

        citations = list(parse_citations(text, display_lookup=DISPLAY_LOOKUP))

        self.assertEqual([c["doc_id"] for c in citations], ["Austin_2008", None])
        self.assertEqual(citations[1]["label"], "Unknown")

    def test_group_label_accepts_ampersand_connector(self):
        text = "Claim (Austin 2008,p.1; North & Weingast 1989,p.2)."

        citations = list(parse_citations(text, display_lookup=DISPLAY_LOOKUP))

        self.assertEqual(
            [(c["doc_id"], c["raw"]) for c in citations],
            [
                ("Austin_2008", "Austin 2008,p.1"),
                ("North&Weingast_1989", "North & Weingast 1989,p.2"),
            ],
        )

    def test_group_labels_accept_apostrophes_and_surname_particles(self):
        for apostrophe in ("'", "\u2019"):
            with self.subTest(apostrophe=apostrophe):
                label = f"O{apostrophe}Brien and van Zanden"
                text = f"Claim ({label} 2009,p.2; Austin 2008,p.1)."

                citations = list(
                    parse_citations(text, display_lookup=DISPLAY_LOOKUP)
                )

                self.assertEqual(
                    [c["doc_id"] for c in citations],
                    ["OBrien&vanZanden_2009", "Austin_2008"],
                )
                self.assertEqual(citations[0]["label"], label)

    def test_group_can_span_newlines(self):
        text = "Claim (Austin 2008,p.1;\n North 1989,p.2)."

        citations = list(parse_citations(text, display_lookup=DISPLAY_LOOKUP))

        self.assertEqual(
            [(c["doc_id"], c["raw"]) for c in citations],
            [
                ("Austin_2008", "Austin 2008,p.1"),
                ("North_1989", "North 1989,p.2"),
            ],
        )

    def test_canonical_group_allows_whitespace_around_colon(self):
        text = "Claim (Austin_2008 : p.1; North_1989\t:\t p.2)."

        citations = list(parse_citations(text))

        self.assertEqual(
            [(c["doc_id"], c["page"], c["raw"]) for c in citations],
            [
                ("Austin_2008", 1, "Austin_2008 : p.1"),
                ("North_1989", 2, "North_1989\t:\t p.2"),
            ],
        )

    def test_manifest_counts_group_members_and_distinct_documents(self):
        text = "Claim (Austin 2008, p.1; North 1989, p.2)."

        rendered, manifest = _emit_citations_manifest(
            text,
            {"Austin_2008", "North_1989"},
            DISPLAY_LOOKUP,
            {"Austin_2008": "C:/docs/austin.pdf", "North_1989": "C:/docs/north.pdf"},
            {},
            {},
            linkify=False,
        )

        self.assertEqual(rendered, text)
        self.assertEqual(len(manifest["citations"]), 2)
        self.assertEqual(manifest["distinct_docs"], 2)
        self.assertEqual(
            [citation["cite_text"] for citation in manifest["citations"]],
            ["Austin 2008, p.1", "North 1989, p.2"],
        )

    def test_manifest_linkifies_each_group_member(self):
        text = "Claim (Austin 2008, p.1; North 1989, p.2)."

        rendered, manifest = _emit_citations_manifest(
            text,
            {"Austin_2008", "North_1989"},
            DISPLAY_LOOKUP,
            {"Austin_2008": "C:/docs/austin.pdf", "North_1989": "C:/docs/north.pdf"},
            {},
            {},
            linkify=True,
        )

        self.assertEqual(len(manifest["citations"]), 2)
        self.assertEqual(manifest["distinct_docs"], 2)
        self.assertIn(
            "([Austin 2008, p.1](file:///C:/docs/austin.pdf#page=1); "
            "[North 1989, p.2](file:///C:/docs/north.pdf#page=2))",
            rendered,
        )

    def test_manifest_preserves_valid_member_beside_malformed_member(self):
        text = "Claim (Austin 2008,p.1; p.3)."

        rendered, manifest = _emit_citations_manifest(
            text,
            {"Austin_2008"},
            DISPLAY_LOOKUP,
            {"Austin_2008": "C:/docs/austin.pdf"},
            {},
            {},
            linkify=True,
        )

        self.assertEqual(len(manifest["citations"]), 1)
        self.assertEqual(manifest["distinct_docs"], 1)
        self.assertIn(
            "([Austin 2008,p.1](file:///C:/docs/austin.pdf#page=1); p.3)",
            rendered,
        )
        self.assertTrue(rendered.endswith("."))


if __name__ == "__main__":
    unittest.main()
