import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from rrr.writer import (
    _build_allowed_citations,
    _build_call_contract,
    _build_evidence_id_map,
    _format_doc_entry,
    _format_outline_block,
    _group_evidence_ids_by_doc,
    _list_allowed_citations,
    _remove_invalid_citations,
    _render_evidence_id_citations,
    _rewrite_style_violations,
    _select_call_evidence,
)


def _doc(index: int, quote_count: int = 4):
    doc_id = f"Author{index}_2000"
    return {
        "doc_id": doc_id,
        "claim": f"Claim for document {index}",
        "quotes": [
            {
                "evidence_id": f"E{index * 100 + q:04d}",
                "doc_id": doc_id,
                "page": q + 1,
                "text": f"Passage {q + 1} from document {index}",
            }
            for q in range(quote_count)
        ],
    }


class WriterEvidenceContractTests(unittest.TestCase):
    def test_invalid_display_citation_with_spaced_page_is_removed(self):
        original = (
            "A supported sentence (Nunn 2008, p.19). "
            "A secondary source is misattributed (Nunn 2008, p. 200)."
        )
        cleaned, removed = _remove_invalid_citations(
            original,
            {"Nunn_2008"},
            allowed_pairs={("Nunn_2008", 19)},
            display_lookup={("nunn", "2008"): "Nunn_2008"},
        )

        self.assertEqual("A supported sentence (Nunn 2008, p.19).", cleaned)
        self.assertEqual(1, len(removed))
        self.assertEqual("invalid_page", removed[0]["reason"])
        self.assertEqual(200, removed[0]["page"])

    def test_packet_bounds_every_evidence_surface(self):
        with patch.dict(os.environ, {"RRR_WRITER_QUOTES_PER_DOC": "2"}):
            packet = _select_call_evidence([_doc(1), _doc(2)])

        pairs, _, pages = _build_allowed_citations(packet)
        allowed = _list_allowed_citations(packet, pages)
        evidence = "\n\n".join(_format_doc_entry(d) for d in packet)
        prompt = f"ALLOWED CITATIONS:\n{allowed}\n\nEvidence:\n{evidence}"
        contract = _build_call_contract("test", packet, allowed, prompt)

        self.assertEqual(4, len(_build_evidence_id_map(packet)))
        self.assertEqual(4, len(pairs))
        self.assertTrue(all(contract["invariants"].values()))
        self.assertNotIn("E0102", allowed)
        self.assertNotIn("E0102", evidence)

    def test_allowed_list_has_no_independent_cap(self):
        packet = _select_call_evidence(
            [_doc(i, quote_count=2) for i in range(1, 21)],
            quotes_per_doc=2,
        )
        _, _, pages = _build_allowed_citations(packet)
        allowed = _list_allowed_citations(packet, pages)
        self.assertEqual(40, allowed.count("  - [E"))

    def test_selected_passage_is_not_truncated(self):
        doc = _doc(1, quote_count=2)
        doc["quotes"][0]["text"] = "a" * 600
        packet = _select_call_evidence([doc], quotes_per_doc=2)
        _, _, pages = _build_allowed_citations(packet)
        allowed = _list_allowed_citations(packet, pages)
        evidence = _format_doc_entry(packet[0])
        prompt = f"{allowed}\n\n{evidence}"
        contract = _build_call_contract("test", packet, allowed, prompt)
        self.assertIn("a" * 600, evidence)
        self.assertTrue(
            contract["invariants"]["displayed_passage_texts_present"]
        )

    def test_unknown_id_cannot_be_rendered(self):
        rendered, stats = _render_evidence_id_citations(
            "A claim [E9999].",
            {"E0001": {"doc_id": "Author1_2000", "page": 1}},
        )
        self.assertEqual("A claim .", rendered)
        self.assertEqual(1, stats["unknown_eids"])
        self.assertEqual(0, stats["replacements"])

    def test_contract_rejects_listed_id_without_displayed_passage(self):
        packet = _select_call_evidence([_doc(1)], quotes_per_doc=2)
        _, _, pages = _build_allowed_citations(packet)
        allowed = _list_allowed_citations(packet, pages)
        contaminated = allowed + "\n  - [E0102] (paper Author1_2000, page 3)"
        with self.assertRaises(AssertionError):
            _build_call_contract("test", packet, contaminated, contaminated)

    def test_outline_ids_come_only_from_the_call_packet(self):
        packet = _select_call_evidence(
            [_doc(1), _doc(3)],
            quotes_per_doc=2,
        )
        block = _format_outline_block(
            {
                "doc_ids": [
                    "Author1_2000",
                    "Author2_2000",
                    "Author3_2000",
                ],
                "elaboration": "The selected sources share a common result.",
            },
            _group_evidence_ids_by_doc(packet),
        )
        self.assertIn("[E0100]", block)
        self.assertIn("[E0300]", block)
        self.assertNotIn("[E0200]", block)
        self.assertNotIn("[E0102]", block)

    def test_style_rewrite_captures_exact_prompt_and_response(self):
        fake_ollama = SimpleNamespace(
            chat=lambda **kwargs: {
                "message": {
                    "content": '{"rewritten":["A direct claim."]}'
                }
            }
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch.dict(
                    os.environ,
                    {"RRR_WRITER_PROMPT_DUMP_DIR": temp_dir},
                ),
                patch.dict(sys.modules, {"ollama": fake_ollama}),
            ):
                rewritten, count, reason = _rewrite_style_violations(
                    ["A claim — with an aside."],
                    [(0, "A claim — with an aside.", ["em_dash"])],
                )
            prompt_path = Path(temp_dir) / "style_rewrite_00.txt"
            response_path = Path(temp_dir) / "style_rewrite_00_response.txt"
            self.assertEqual(["A direct claim."], rewritten)
            self.assertEqual(1, count)
            self.assertEqual("ok", reason)
            self.assertIn(
                "You revise academic prose. Be concise. Return JSON only.",
                prompt_path.read_text(encoding="utf-8"),
            )
            self.assertEqual(
                '{"rewritten":["A direct claim."]}',
                response_path.read_text(encoding="utf-8"),
            )

    def test_style_rewrite_rejects_parenthetical_citation_drift(self):
        fake_ollama = SimpleNamespace(
            chat=lambda **kwargs: {
                "message": {
                    "content": (
                        '{"rewritten":["A direct claim '
                        '(Acemoglu et al. 2001, p.33)."]}'
                    )
                }
            }
        )
        original = (
            "A claim — with an aside "
            "(Acemoglu et al. 2002, p.33)."
        )
        with patch.dict(sys.modules, {"ollama": fake_ollama}):
            rewritten, count, reason = _rewrite_style_violations(
                [original],
                [(0, original, ["em_dash"])],
            )
        self.assertEqual([original], rewritten)
        self.assertEqual(0, count)
        self.assertEqual("ok", reason)

    def test_style_rewrite_accepts_preserved_parenthetical_citation(self):
        fake_ollama = SimpleNamespace(
            chat=lambda **kwargs: {
                "message": {
                    "content": (
                        '{"rewritten":["A direct claim '
                        '(Acemoglu et al. 2002, p.33)."]}'
                    )
                }
            }
        )
        original = (
            "A claim — with an aside "
            "(Acemoglu et al. 2002, p.33)."
        )
        with patch.dict(sys.modules, {"ollama": fake_ollama}):
            rewritten, count, reason = _rewrite_style_violations(
                [original],
                [(0, original, ["em_dash"])],
            )
        self.assertEqual(
            ["A direct claim (Acemoglu et al. 2002, p.33)."],
            rewritten,
        )
        self.assertEqual(1, count)
        self.assertEqual("ok", reason)

    def test_style_rewrite_rejects_duplicated_citation(self):
        fake_ollama = SimpleNamespace(
            chat=lambda **kwargs: {
                "message": {
                    "content": (
                        '{"rewritten":["A direct claim '
                        '(Nunn 2008, p.3) (Nunn 2008, p.3)."]}'
                    )
                }
            }
        )
        original = "A claim — with an aside (Nunn 2008, p.3)."
        with patch.dict(sys.modules, {"ollama": fake_ollama}):
            rewritten, count, reason = _rewrite_style_violations(
                [original],
                [(0, original, ["em_dash"])],
            )
        self.assertEqual([original], rewritten)
        self.assertEqual(0, count)
        self.assertEqual("ok", reason)

    def test_style_rewrite_rejects_non_string_array_member(self):
        fake_ollama = SimpleNamespace(
            chat=lambda **kwargs: {
                "message": {
                    "content": (
                        '{"rewritten":[{"sentence":"A direct claim."}]}'
                    )
                }
            }
        )
        original = "A claim — with an aside."
        with patch.dict(sys.modules, {"ollama": fake_ollama}):
            rewritten, count, reason = _rewrite_style_violations(
                [original],
                [(0, original, ["em_dash"])],
            )
        self.assertEqual([original], rewritten)
        self.assertEqual(0, count)
        self.assertEqual("ok", reason)


if __name__ == "__main__":
    unittest.main()
