"""The quotation verifier scans curly quotation marks and spares coverage quotations (vlm.2.5)."""
import pytest

from rrr import writer
from rrr.render import _build_display_lookup

DOCS = {"McCloskey_2015", "Mokyr_2005"}
PAGE_10 = "The Dutch and the French were, in the older view, trading nations of no great dignity."
PAGE_9 = "The bourgeois and aristocratically dishonorable countries, the Netherlands and France, embraced the new rhetoric."
LEAD = "Rhetoric about trade changed first (Mokyr 2005, p.49). "
TAIL = " Mokyr (2005, p.49) places the same shift in the culture of useful knowledge."


@pytest.fixture
def pages(tmp_path, monkeypatch):
    for doc, page, text in (("McCloskey_2015", 10, PAGE_10), ("McCloskey_2015", 9, PAGE_9), ("Mokyr_2005", 49, "useful knowledge")):
        (tmp_path / f"{doc}_page_{page}.txt").write_text(text, encoding="utf-8")
    import rrr.paths

    monkeypatch.setattr(
        rrr.paths, "page_text_path",
        lambda doc_id, page: tmp_path / f"{doc_id}_page_{int(page)}.txt",
    )
    return tmp_path


@pytest.fixture
def lookup():
    return _build_display_lookup(DOCS)


def test_curly_quotation_on_the_wrong_page_is_stripped(pages, lookup):
    sentence = ("McCloskey (2015, p.10) illustrates how the “bourgeois and aristocratically dishonorable” "
                "countries embraced new rhetoric.")
    out, stats = writer._strip_fabricated_quotes(LEAD + sentence + TAIL, DOCS, display_lookup=lookup)
    assert stats["checked_quotes"] == 1
    assert stats["fabricated_stripped"] == 1
    assert "dishonorable" not in out
    assert out.startswith(LEAD.rstrip())
    assert "Mokyr (2005, p.49) places" in out


def test_curly_quotation_on_the_right_page_is_kept(pages, lookup):
    sentence = ("McCloskey (2015, p.9) calls them the “bourgeois and aristocratically dishonorable” "
                "countries of the age.")
    text = LEAD + sentence + TAIL
    out, stats = writer._strip_fabricated_quotes(text, DOCS, display_lookup=lookup)
    assert out == text
    assert stats["verified_real"] == 1


def test_coverage_quotation_with_sentinel_is_not_checked(pages, lookup):
    patch = (writer._COVERAGE_PATCH_SENTINEL + "McCloskey (2015, p.10) reports, "
             "“bourgeois and aristocratically dishonorable countries embraced the new rhetoric”")
    text = LEAD + patch
    out, stats = writer._strip_fabricated_quotes(text, DOCS, display_lookup=lookup)
    assert out == text
    assert stats["checked_quotes"] == 0
    assert stats["coverage_quotes_skipped"] == 1


def test_coverage_quotation_by_template_is_not_checked(pages, lookup):
    patch = ("McCloskey (2015, p.10) reports, "
             "“bourgeois and aristocratically dishonorable countries embraced the new rhetoric” "
             "Mokyr (2005, p.49) reports, “the growth of useful knowledge was the mainspring of it all”")
    text = LEAD + patch
    out, stats = writer._strip_fabricated_quotes(text, DOCS, display_lookup=lookup)
    assert out == text
    assert stats["checked_quotes"] == 0
    assert stats["coverage_quotes_skipped"] == 2


def test_straight_quotation_behaviour_is_unchanged(pages, lookup):
    sentence = 'McCloskey (2015, p.10) writes of "bourgeois and aristocratically dishonorable" countries.'
    out, stats = writer._strip_fabricated_quotes(LEAD + sentence + TAIL, DOCS, display_lookup=lookup)
    assert stats["checked_quotes"] == 1
    assert stats["fabricated_stripped"] == 1
    assert "dishonorable" not in out
