"""The quotation verifier never ships a failed quotation as a quotation (vlm.2.4)."""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import rrr.paths
from rrr import writer
from rrr.render import _build_display_lookup

DOCS = {"Mokyr_2005", "Kuznets_1973"}
PAGE_49 = "None of these results are materially different whether we count journals by first appearance only."
PAGE_44 = "Economic historians have long held that the growth of useful knowledge drives modern growth."
PAGES = (
    ("Mokyr_2005", 49, PAGE_49),
    ("Mokyr_2005", 44, PAGE_44),
    ("Kuznets_1973", 1, "x"),
    ("Kuznets_1973", 6, "y"),
)
QUOTE_SENTENCE = (
    'The accumulation of useful knowledge, which Mokyr refers to as '
    '"the growth of useful knowledge" (Mokyr 2005, p.49), is the mainspring of sustained economic growth.'
)
CITED_LEAD = (
    "Technological progress is the primary driver of long-run economic growth, as Mokyr and Kuznets "
    "(Kuznets 1973, p.1; Mokyr 2005, p.49) both argue. "
)
UNCITED_LEAD = "Technological progress is the primary driver of long-run economic growth, as Mokyr argues. "
TAIL = " Kuznets (1973, p.6) adds that such change demands institutional adjustment."


class QuoteGuardTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        for doc, page, text in PAGES:
            (root / f"{doc}_page_{page}.txt").write_text(text, encoding="utf-8")
        patcher = mock.patch.object(
            rrr.paths, "page_text_path",
            lambda doc_id, page: root / f"{doc_id}_page_{int(page)}.txt",
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        self.lookup = _build_display_lookup(DOCS)

    def test_cited_surname_is_not_stranded(self):
        assert writer._drop_would_strand_surname(
            CITED_LEAD, QUOTE_SENTENCE, DOCS, display_lookup=self.lookup) is False

    def test_uncited_surname_is_stranded(self):
        assert writer._drop_would_strand_surname(
            UNCITED_LEAD, QUOTE_SENTENCE, DOCS, display_lookup=self.lookup) is True

    def test_failed_quotation_beside_a_cited_surname_is_stripped(self):
        out, stats = writer._strip_fabricated_quotes(
            CITED_LEAD + QUOTE_SENTENCE + TAIL, DOCS, display_lookup=self.lookup)
        assert stats["checked_quotes"] == 1
        assert stats["fabricated_stripped"] == 1
        assert stats.get("strand_guard_kept", 0) == 0
        assert "growth of useful knowledge" not in out
        assert out.startswith(CITED_LEAD.rstrip())
        assert "Kuznets (1973, p.6)" in out

    def test_failed_quotation_beside_an_uncited_surname_loses_its_marks(self):
        out, stats = writer._strip_fabricated_quotes(
            UNCITED_LEAD + QUOTE_SENTENCE + TAIL, DOCS, display_lookup=self.lookup)
        assert stats["fabricated_stripped"] == 0
        assert stats["strand_guard_kept"] == 1
        assert stats["strand_guard_dequoted"] == 1
        assert stats["fabrications"][0]["action"] == "quotation_marks_removed"
        assert "Mokyr refers to as the growth of useful knowledge (Mokyr 2005, p.49)" in out
        assert not writer._FABRICATED_QUOTE_RE.search(out)
        assert "Kuznets (1973, p.6)" in out

    def test_verified_quotation_is_untouched(self):
        text = UNCITED_LEAD + (
            'Mokyr writes that "the growth of useful knowledge drives modern growth" (Mokyr 2005, p.44).')
        out, stats = writer._strip_fabricated_quotes(text, DOCS, display_lookup=self.lookup)
        assert out == text
        assert stats["verified_real"] == 1
        assert stats["fabricated_stripped"] == 0

    def test_two_failed_quotations_in_one_sentence_strip_it_once(self):
        sentence = (
            'Mokyr calls this "the growth of useful knowledge in Europe" and also '
            '"a change in the social basis of knowledge" (Mokyr 2005, p.49).')
        out, stats = writer._strip_fabricated_quotes(
            CITED_LEAD + sentence + TAIL, DOCS, display_lookup=self.lookup)
        assert stats["checked_quotes"] == 2
        assert stats["fabricated_stripped"] == 1
        assert "social basis" not in out
        assert "Kuznets (1973, p.6)" in out
        assert out.startswith(CITED_LEAD.rstrip())


if __name__ == "__main__":
    unittest.main()
