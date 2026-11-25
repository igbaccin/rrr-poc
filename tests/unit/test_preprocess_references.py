import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from _pdf_fixtures import make_layout_pdf

import rrr.preprocess as preprocess
from rrr.preprocess import (
    _find_reference_start,
    _is_strong_reference_header,
    _truncate_at_reference_header,
    extract_pages,
    scan_reference_leaks,
)


BODY = (
    "The basic needs approach emphasizes increased productivity and the growth "
    "of commodities consumed by the poor. A combination of regional growth "
    "strategies and redistribution towards the northern savanna would narrow "
    "the gap in living standards that the colonial economy left behind. "
)
BODY_LEFT = (
    "Left column. " + BODY + "The conflict between growth and basic needs is "
    "sharpest where public spending is thin and markets are distant."
)
BODY_RIGHT = (
    "Right column. Regional policy analysis, addressed to the management of "
    "the impact of national policy, would direct investment to the districts "
    "with the lowest indices of welfare. " + BODY + "This closes the argument."
)
ENTRIES_LEFT = (
    "BENNETT, D. (1980), The Future of Welfare Geography. The Canadian Geographer 24.\n"
    "DARKOH, M. B. (1976), Recent Trends in Regional Planning in Ghana. Journal of Geography 12.\n"
    "DICKSON, K. B. (1970), The Population Factor in the Economic Development of Ghana.\n"
    "EWUSI, K. (1976), Disparities in Levels of Regional Development in Ghana. Legon, Accra.\n"
    "FORDE, E. R. (1968), The Population of Ghana: A Study of the Spatial Relationships.\n"
    "GAISIE, S. K. (1974), Population Growth and its Components in Ghana. Legon, Accra."
)
ENTRIES_RIGHT = (
    "HARVEY, D. (1973), Social Justice and the City. New York: Edward Arnold.\n"
    "LARSON, D. (1979), The PQLI: Measuring the Condition of the World's Poor. New York.\n"
    "MORRIS, M. D. (1979), Measuring the Condition of the World's Poor. New York: Pergamon.\n"
    "RAWLS, J. (1971), A Theory of Justice. Cambridge, Mass.: Harvard University Press.\n"
    "SMITH, D. M. (1979), Where the Grass is Greener: Living in an Unequal World. London.\n"
    "STREETEN, P. P. (1981), First Things First: Meeting Basic Human Needs. Oxford."
)
PLAIN_PAGE = [((40, 30, 430, 640), ("Body page. " + BODY) * 4)]

LEFT = (38, 24, 232, 232)
RIGHT = (241, 24, 436, 232)
HEADING = (40, 250, 232, 266)
LIST_LEFT = (40, 270, 232, 560)
LIST_RIGHT = (241, 270, 436, 560)
FOOTER = (41, 640, 436, 654)


def _process(pdf: Path, root: Path, doc_id: str = "Doc_1988") -> dict:
    def data_path(*parts):
        return root.joinpath("data", *parts)

    with patch.object(preprocess, "data_path", data_path):
        result = preprocess._process_one({"doc_id": doc_id, "pdf_path": str(pdf)})
    assert result["ok"], result
    pages = {}
    for path in (root / "data" / "page_text").glob(f"{doc_id}_page_*.txt"):
        pages[int(path.stem.rsplit("_", 1)[1])] = path.read_text(encoding="utf-8")
    return pages


class HeadingPunctuationTests(unittest.TestCase):
    def test_a_heading_may_close_with_a_colon_or_a_full_stop(self):
        for heading in (
            "References:", "References.", "REFERENCES:", "References :",
            "Bibliography:", "BIBLIOGRAPHY.", "Works Cited:", "Reference List:",
            "Literature Cited.", "Sources:", "References",
        ):
            page = BODY + "\n" + heading + " \n" + ENTRIES_LEFT
            self.assertTrue(_is_strong_reference_header(page), heading)
            kept, cut = _truncate_at_reference_header(page)
            self.assertTrue(cut, heading)
            self.assertEqual(BODY.rstrip(), kept, heading)

    def test_the_colon_form_is_seen_by_the_leak_invariant(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            page = Path(temp_dir) / "Doc_1988_page_8.txt"
            page.write_text(BODY + "\nReferences:\n" + ENTRIES_LEFT, encoding="utf-8")
            leaks = scan_reference_leaks(temp_dir)
        self.assertEqual(1, len(leaks))
        self.assertIn("standalone ref header", leaks[0][1])

    def test_the_word_in_running_text_is_no_heading(self):
        for prose in (
            "References: see Appendix B for the complete list of sources used.",
            "The references cited in this section are discussed in turn below.",
            "Several references. Each one is taken up again in the conclusion.",
            "Sources: Ghana Statistical Service (2000) and own calculations.",
            "Bibliography entries of this kind are rare in the colonial files.",
            "References to migrant labour recur throughout the annual reports.",
        ):
            page = BODY + "\n" + prose + "\n" + BODY
            self.assertFalse(_is_strong_reference_header(page), prose)

    def test_running_text_does_not_cut_a_page_or_move_the_boundary(self):
        mention = (
            BODY + "\nReferences: see Appendix B for the complete list.\n" + BODY
            + "\nThe references cited above are discussed in turn.\n" + BODY
        )
        pages = [BODY * 3, mention, mention, BODY * 3,
                 BODY + "\nReferences:\n" + ENTRIES_LEFT]
        self.assertEqual(4, _find_reference_start(pages))

    def test_a_glued_capital_heading_with_a_colon_is_cut(self):
        page = BODY.rstrip() + "REFERENCES:Alatas, S. (1977), The Myth of the Lazy Native."
        self.assertTrue(_is_strong_reference_header(page))
        kept, cut = _truncate_at_reference_header(page)
        self.assertTrue(cut)
        self.assertEqual(BODY.rstrip(), kept)


class TwoColumnReferencePageTests(unittest.TestCase):
    def _pdf(self, root, last_page):
        return make_layout_pdf(
            root / "paper.pdf", [PLAIN_PAGE, PLAIN_PAGE, PLAIN_PAGE, last_page],
        )

    def test_the_list_underneath_both_columns_keeps_both_columns_of_text(self):
        last_page = [
            (LEFT, BODY_LEFT),
            (HEADING, "References:"),
            (LIST_LEFT, ENTRIES_LEFT),
            (RIGHT, BODY_RIGHT),
            (LIST_RIGHT, ENTRIES_RIGHT),
            (FOOTER, "38 Tijdschrift voor Econ. en Soc. Geografie 79 (1988) Nr. 1"),
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pdf = self._pdf(root, last_page)
            stream = extract_pages(str(pdf))[3]
            pages = _process(pdf, root)
            leaks = scan_reference_leaks(root / "data" / "page_text")
        self.assertLess(stream.index("References:"), stream.index("Right column."))
        self.assertEqual([1, 2, 3, 4], sorted(pages))
        kept = " ".join(pages[4].split())
        self.assertIn("Left column.", kept)
        self.assertIn("Right column.", kept)
        self.assertIn("This closes the argument.", kept)
        self.assertLess(kept.index("Left column."), kept.index("Right column."))
        for word in ("References", "BENNETT", "HARVEY", "STREETEN", "Tijdschrift"):
            self.assertNotIn(word, kept)
        self.assertEqual([], leaks)

    def test_a_list_that_flows_into_the_right_column_adds_nothing(self):
        last_page = [
            (LEFT, BODY_LEFT),
            (HEADING, "References:"),
            (LIST_LEFT, ENTRIES_LEFT),
            ((241, 24, 436, 560), ENTRIES_RIGHT + "\n" + ENTRIES_LEFT),
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pdf = self._pdf(root, last_page)
            stream_cut, cut = _truncate_at_reference_header(extract_pages(str(pdf))[3])
            pages = _process(pdf, root)
        self.assertTrue(cut)
        self.assertEqual(stream_cut, pages[4])
        self.assertNotIn("HARVEY", pages[4])

    def test_a_single_column_page_is_written_exactly_as_the_stream_cut(self):
        for heading in ("References:", "References"):
            last_page = [
                ((40, 24, 430, 232), BODY_LEFT + " " + BODY_RIGHT),
                ((40, 250, 430, 266), heading),
                ((40, 270, 430, 560), ENTRIES_LEFT + "\n" + ENTRIES_RIGHT),
                (FOOTER, "38 Tijdschrift voor Econ. en Soc. Geografie 79 (1988) Nr. 1"),
            ]
            with tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                pdf = self._pdf(root, last_page)
                stream_cut, _ = _truncate_at_reference_header(extract_pages(str(pdf))[3])
                pages = _process(pdf, root)
            self.assertEqual(stream_cut, pages[4], heading)
            self.assertIn("This closes the argument.", " ".join(pages[4].split()))

    def test_a_short_block_beside_the_heading_is_not_taken_for_body_text(self):
        last_page = [
            (LEFT, BODY_LEFT),
            (HEADING, "References:"),
            (LIST_LEFT, ENTRIES_LEFT),
            ((241, 24, 436, 40), "Regional inequalities in Ghana"),
            (LIST_RIGHT, ENTRIES_RIGHT),
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pdf = self._pdf(root, last_page)
            stream_cut, _ = _truncate_at_reference_header(extract_pages(str(pdf))[3])
            pages = _process(pdf, root)
        self.assertEqual(stream_cut, pages[4])

    def test_a_document_without_a_mid_page_heading_is_written_as_extracted(self):
        list_page = [((40, 24, 430, 40), "References"),
                     ((40, 50, 430, 600), ENTRIES_LEFT + "\n" + ENTRIES_RIGHT)]
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pdf = make_layout_pdf(
                root / "paper.pdf", [PLAIN_PAGE, PLAIN_PAGE, PLAIN_PAGE, list_page],
            )
            extracted = extract_pages(str(pdf))
            pages = _process(pdf, root)
        self.assertEqual({1: extracted[0], 2: extracted[1], 3: extracted[2]}, pages)


if __name__ == "__main__":
    unittest.main()
