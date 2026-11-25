import inspect
import unittest

import rrr.writer as writer
from rrr import outline


HOSTED = hasattr(writer, "_WRITER_HOST_EXECUTION_PROFILE")

PLAN = {
    "topic_shape": "explanatory",
    "topic_cause": "",
    "topic_outcome": "decline of honeybee colonies",
    "ordered_cluster_ids": ["C3", "C1", "C2", "C4"],
    "clusters": [
        {"cluster_id": "C1", "relation": "explains_outcome", "doc_ids": ["a", "b", "c"],
         "shared_cause": "neonicotinoid pesticide exposure", "shared_thread": "pesticides and colony health"},
        {"cluster_id": "C2", "relation": "explains_outcome", "doc_ids": ["d"],
         "shared_cause": "varroa mite infestation", "shared_thread": "parasites and viruses"},
        {"cluster_id": "C3", "relation": "documents_outcome", "doc_ids": ["e", "f"],
         "shared_cause": "", "shared_thread": "measured colony losses"},
        {"cluster_id": "C4", "relation": "disputes_outcome", "doc_ids": ["g"],
         "shared_cause": "", "shared_thread": "stable colony counts in Asia"},
    ],
}


def stream_prompt(relation, explanation=""):
    kwargs = {"outline_block": "BRIEF", "claims_so_far": "", "relation": relation,
              "topic_shape": "explanatory", "explanation": explanation}
    if not HOSTED:
        kwargs.update(lead_surname="Abel", forbidden_opener_surnames=["Cruz"])
    return writer._build_stream_prompt(
        "Why are honeybee colonies declining?", "pesticides and colony health", "EVIDENCE",
        "ALLOWED", "The previous section ended here.", **kwargs)


class WriterOpenQuestionTests(unittest.TestCase):
    def test_every_relation_stage_two_can_return_has_a_gloss(self):
        for shape, relations in outline._RELATIONS_BY_SHAPE.items():
            for relation in relations:
                with self.subTest(shape=shape, relation=relation):
                    self.assertTrue(writer._RELATION_OPENERS.get(relation))

    def test_the_new_glosses_place_a_stream_against_the_outcome(self):
        self.assertIn("EXPLANATION of the outcome", writer._RELATION_OPENERS["explains_outcome"])
        self.assertIn("proposes no cause", writer._RELATION_OPENERS["documents_outcome"])
        self.assertIn("premise", writer._RELATION_OPENERS["disputes_outcome"])
        for relation in ("explains_outcome", "documents_outcome", "disputes_outcome"):
            self.assertNotIn("topic's causal variable", writer._RELATION_OPENERS[relation])

    def test_the_opening_is_given_the_explanations_on_offer(self):
        summary = writer._explanation_summary(PLAN, 7)
        self.assertEqual(
            "Topic shape: explanatory. The topic asks what explains decline of honeybee colonies "
            "and names no cause. Across 7 sources the literature offers these explanations: "
            "neonicotinoid pesticide exposure (3 source(s)); varroa mite infestation (1 source(s)). "
            "Set them side by side as answers to the same question. Add no explanation that is "
            "absent from this list. 2 source(s) establish the outcome itself (its size, timing or "
            "place). 1 source(s) dispute that the outcome holds as the topic states it.",
            summary)

    def test_the_explanations_keep_the_order_stage_three_chose(self):
        plan = dict(PLAN, ordered_cluster_ids=["C2", "C1", "C3", "C4"])
        summary = writer._explanation_summary(plan, 7)
        self.assertLess(summary.index("varroa"), summary.index("neonicotinoid"))

    def test_a_corpus_without_an_explanation_is_described_as_such(self):
        plan = {"topic_outcome": "glacier surges", "clusters": [
            {"cluster_id": "C1", "relation": "documents_outcome", "doc_ids": ["a", "b"]}]}
        summary = writer._explanation_summary(plan, 2)
        self.assertIn("No stream among the 2 sources proposes a cause.", summary)
        self.assertNotIn("offers these explanations", summary)

    def test_an_explanation_without_a_label_takes_the_quoted_cause_then_the_stream_name(self):
        quoted = {"topic_outcome": "glacier surges", "clusters": [
            {"cluster_id": "C1", "relation": "explains_outcome", "doc_ids": ["a"], "shared_cause": "",
             "cluster_cause_quote": "meltwater raised the basal pressure", "shared_thread": "glacier hydrology"}]}
        self.assertIn("meltwater raised the basal pressure (1 source(s))", writer._explanation_summary(quoted, 1))
        bare = {"topic_outcome": "glacier surges", "clusters": [
            {"cluster_id": "C1", "relation": "explains_outcome", "doc_ids": ["a"],
             "shared_cause": "", "shared_thread": "basal water pressure"}]}
        self.assertIn("basal water pressure (1 source(s))", writer._explanation_summary(bare, 1))

    def test_an_explaining_section_is_told_its_explanation(self):
        prompt = stream_prompt("explains_outcome", "neonicotinoid pesticide exposure")
        self.assertIn("This stream's explanation of the outcome: neonicotinoid pesticide exposure.", prompt)
        self.assertIn("(explains_outcome)", prompt)

    def test_other_sections_are_not_given_an_explanation(self):
        for relation in ("documents_outcome", "disputes_outcome", "adjacent", "rival_to_topic_cause"):
            with self.subTest(relation=relation):
                self.assertEqual(stream_prompt(relation), stream_prompt(relation, "some explanation"))
                self.assertNotIn("This stream's explanation", stream_prompt(relation, "some explanation"))

    def test_an_empty_explanation_adds_nothing(self):
        self.assertNotIn("This stream's explanation", stream_prompt("explains_outcome", ""))

    def test_the_review_composer_uses_both_additions(self):
        source = inspect.getsource(writer.compose_from_ledger)
        self.assertIn("_explanation_summary(outline_plan, len(docs))", source)
        self.assertIn('if topic_shape == "explanatory" else ""', source)


if __name__ == "__main__":
    unittest.main()
