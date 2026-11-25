import re
import unittest
from pathlib import Path

from rrr import reasoner


ROOT = Path(__file__).resolve().parents[2]
BATTERY = ROOT / "scripts" / "run_comprehensive_battery.sh"


class BreakdownRecordTests(unittest.TestCase):
    def test_a_breakdown_is_not_a_refusal_in_any_record(self):
        for reason in ("stage0_llm_failed", "stage1_clustering_failed"):
            with self.subTest(reason=reason):
                record = reasoner._breakdown_record(reason, {"failure": "invalid_answer", "detail": '{"topic_shape": "open"}'})
                self.assertIs(False, record["result"]["refusal"])
                self.assertIs(True, record["result"]["breakdown"])
                self.assertEqual(reason, record["result"]["reason"])
                self.assertEqual("invalid_answer", record["result"]["detail"])
                self.assertEqual('{"topic_shape": "open"}', record["result"]["reply_excerpt"])
                self.assertIn("fault of the run", record["result"]["explanation"])
                self.assertEqual({"refusal": False, "breakdown": True, "breakdown_reason": reason,
                                  "breakdown_detail": "invalid_answer"}, record["values"])
                self.assertEqual({"breakdown": reason, "breakdown_detail": "invalid_answer"}, record["extra"])
                self.assertNotIn("refusal", record["extra"])
                self.assertNotIn("refusal_reason", record["values"])

    def test_the_console_line_cannot_be_read_as_a_refusal(self):
        for diagnostics in (None, {}, {"failure": "model_call_error", "detail": "RuntimeError: refused"}):
            line = reasoner._breakdown_record("stage0_llm_failed", diagnostics)["console"]
            self.assertTrue(line.startswith("[Layered-T2] breakdown=stage0_llm_failed"))
            self.assertNotIn("refusal=", line)
            self.assertNotIn("refusing to run", line)
            self.assertIsNone(re.search(r"\bREFUS", line))

    def test_the_reason_can_be_read_back_from_the_console_line(self):
        line = reasoner._breakdown_record("stage1_clustering_failed", {"failure": "no_valid_cluster_plan"})["console"]
        self.assertEqual("stage1_clustering_failed", re.search(r"breakdown=([a-z0-9_]+)", line).group(1))

    def test_a_breakdown_without_diagnostics_still_explains_itself(self):
        record = reasoner._breakdown_record("stage0_llm_failed")
        self.assertEqual("", record["result"]["detail"])
        self.assertNotIn("reply_excerpt", record["result"])
        self.assertIn("Stage 0", record["result"]["explanation"])

    @unittest.skipUnless(BATTERY.is_file(), "the battery script lives in the private repository")
    def test_the_battery_keeps_counting_a_breakdown_as_a_structured_failure(self):
        text = BATTERY.read_text(encoding="utf-8")
        breakdown = text.index('grep -q "breakdown="')
        refusal = text.index('grep -q "refusal="')
        self.assertLess(breakdown, refusal, "the breakdown branch must come first")
        branch = text[breakdown:refusal]
        self.assertIn('\\"refusal\\": true, \\"reason\\": \\"stage\\"', branch)
        self.assertIn('\\"breakdown\\": true', branch)
        self.assertIn("breakdown_reason", branch)


class OpenQuestionVerdictTests(unittest.TestCase):
    PLAN = {
        "topic_shape": "explanatory",
        "topic_outcome": "decline of honeybee colonies",
        "clusters": [
            {"cluster_id": "C1", "relation": "explains_outcome", "doc_ids": ["a", "b", "c"],
             "shared_cause": "neonicotinoid pesticide exposure", "shared_thread": "pesticides"},
            {"cluster_id": "C2", "relation": "explains_outcome", "doc_ids": ["d"],
             "shared_cause": "", "shared_thread": "parasites and viruses"},
            {"cluster_id": "C3", "relation": "documents_outcome", "doc_ids": ["e", "f"], "shared_cause": ""},
            {"cluster_id": "C4", "relation": "disputes_outcome", "doc_ids": ["g"], "shared_cause": ""},
            {"cluster_id": "C5", "relation": "adjacent", "doc_ids": ["h"], "shared_cause": ""},
        ],
    }

    def test_the_verdict_lists_the_explanations_and_weighs_nothing(self):
        verdict = reasoner._open_question_verdict(self.PLAN)
        self.assertEqual(
            "Open question, no claim to weigh. The corpus offers 2 explanations of decline of "
            "honeybee colonies: neonicotinoid pesticide exposure (3 papers); parasites and viruses (1 paper)",
            verdict["headline"])
        self.assertEqual("explains=4, documents=2, disputes=1, adjacent=1", verdict["breakdown"])
        for word in ("support", "rival", "Well-supported", "Contested"):
            self.assertNotIn(word, verdict["headline"])

    def test_a_corpus_with_no_explanation_says_so(self):
        plan = {"topic_outcome": "glacier surges",
                "clusters": [{"cluster_id": "C1", "relation": "documents_outcome", "doc_ids": ["a"]}]}
        verdict = reasoner._open_question_verdict(plan)
        self.assertIn("No stream of the corpus offers an explanation of glacier surges", verdict["headline"])
        self.assertEqual("explains=0, documents=1, disputes=0, adjacent=0", verdict["breakdown"])

    def test_the_explanatory_relations_have_headings(self):
        relations = [rel for rel, _ in reasoner._OPEN_QUESTION_RELATIONS]
        self.assertEqual(["explains_outcome", "documents_outcome", "disputes_outcome"], relations)
        for _rel, label in reasoner._OPEN_QUESTION_RELATIONS:
            self.assertTrue(label)


if __name__ == "__main__":
    unittest.main()
