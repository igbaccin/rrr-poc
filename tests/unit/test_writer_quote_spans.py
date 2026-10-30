"""Quotation spans never cross marks of the other kind, and a demoted quotation leaves no stray mark (vlm.2.5)."""
import pytest

from rrr import writer
from rrr.render import _build_display_lookup

DOCS = {"AcemogluEtAl_2001", "Mokyr_2005"}
PAGE_27 = "Institutional features, such as expropriation risk, should be read as outcomes of deeper institutions."
PAGE_49 = "Scholarly societies lowered the access costs to new ideas across Europe."


@pytest.fixture
def pages(tmp_path, monkeypatch):
    for doc, page, text in (("AcemogluEtAl_2001", 27, PAGE_27), ("Mokyr_2005", 49, PAGE_49)):
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


def test_stray_straight_mark_does_not_run_into_a_coverage_quotation(pages, lookup):
    text = ('Extractive institutions persist, proving that even "fundamental changes can be undermined by '
            'settlement patterns. Acemoglu et al. (2001, p.27) report, “Institutional features, such as '
            'expropriation risk, should be read as outcomes of deeper institutions.”')
    out, stats = writer._strip_fabricated_quotes(text, DOCS, display_lookup=lookup)
    assert out == text
    assert stats["checked_quotes"] == 0
    assert stats["coverage_quotes_skipped"] == 1


def test_demoted_quotation_leaves_no_stray_mark_in_its_sentence(pages, lookup):
    lead = "Technological progress is the primary driver of growth, as Mokyr argues. "
    sentence = ('Mokyr holds that "the growth of useful knowledge drives everything" and adds "a further point '
                '(Mokyr 2005, p.49).')
    tail = " Kuznets saw the same pattern."
    out, stats = writer._strip_fabricated_quotes(lead + sentence + tail, DOCS, display_lookup=lookup)
    assert stats["strand_guard_dequoted"] == 1
    assert stats["strand_guard_marks_cleared"] == 1
    assert '"' not in out
    assert "(Mokyr 2005, p.49)" in out
    assert "Kuznets saw the same pattern." in out
    assert "the growth of useful knowledge drives everything" in out


def test_balanced_demotion_clears_nothing_else(pages, lookup):
    lead = "Technological progress is the primary driver of growth, as Mokyr argues. "
    sentence = 'Mokyr holds that "the growth of useful knowledge drives everything" (Mokyr 2005, p.49).'
    out, stats = writer._strip_fabricated_quotes(lead + sentence, DOCS, display_lookup=lookup)
    assert stats["strand_guard_dequoted"] == 1
    assert stats.get("strand_guard_marks_cleared", 0) == 0
    assert out.endswith("Mokyr holds that the growth of useful knowledge drives everything (Mokyr 2005, p.49).")
