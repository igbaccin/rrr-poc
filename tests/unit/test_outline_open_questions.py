import hashlib
import json
import os
import sys
import types
import unittest
from unittest import mock

from rrr import outline


BEES = "Why are honeybee colonies declining?"
PRESS = "How did the printing press change literacy in early modern Europe?"
TITLES = [{"doc_id": f"Doc_{i}", "citation": f"Author {i} (200{i}) 'Paper {i}'."} for i in range(6)]

SUMMARIES = [
    {"doc_id": "Abel_2010", "citation": "Abel (2010)", "claim": "The paper argues that neonicotinoid pesticides weakened colonies.",
     "quotes": [{"page": 4, "text": "Colonies exposed to neonicotinoid pesticides lost foragers at twice the rate."}]},
    {"doc_id": "Baum_2012", "citation": "Baum (2012)", "claim": "The paper argues that pesticide exposure impairs homing.",
     "quotes": [{"page": 9, "text": "Sublethal exposure impaired the homing of worker bees."}]},
    {"doc_id": "Cruz_2014", "citation": "Cruz (2014)", "claim": "The paper argues that the varroa mite and the viruses it carries killed colonies.",
     "quotes": [{"page": 2, "text": "Varroa mite infestation preceded every recorded collapse."}]},
    {"doc_id": "Dahl_2016", "citation": "Dahl (2016)", "claim": "The paper measures colony losses across forty countries since 1990.",
     "quotes": [{"page": 1, "text": "Managed colonies fell by a quarter between 1990 and 2010."}]},
]


def reply(obj) -> dict:
    return {"message": {"content": json.dumps(obj)}}


def stage0(shape="causal", fit="PROCEED", cause="", outcome=""):
    return {"topic_shape": shape, "topic_shape_rationale": "r", "corpus_fit": fit,
            "corpus_fit_rationale": "r", "topic_cause": cause, "topic_outcome": outcome}


class FakeModel:
    def __init__(self, replies=None, router=None):
        self.replies = list(replies or [])
        self.router = router
        self.prompts = []

    def chat(self, **kwargs):
        prompt = kwargs["messages"][0]["content"]
        self.prompts.append(prompt)
        if self.router is not None:
            return self.router(prompt)
        item = self.replies.pop(0)
        if isinstance(item, Exception):
            raise item
        if isinstance(item, str):
            return {"message": {"content": item}}
        return reply(item)


class StubbedStages(unittest.TestCase):
    def run_with(self, fake, fn, *args, **kwargs):
        module = types.ModuleType("ollama")
        module.chat = fake.chat
        env = {k: v for k, v in os.environ.items()
               if k not in ("RRR_CORPUS_LANG", "RRR_TOPIC_LANG")}
        with mock.patch.dict(sys.modules, {"ollama": module}), \
                mock.patch.dict(os.environ, env, clear=True), \
                mock.patch.object(outline, "_load_cache", return_value=None), \
                mock.patch.object(outline, "_save_cache", return_value=None):
            return fn(*args, **kwargs)


class Stage0PromptTests(unittest.TestCase):
    def setUp(self):
        self.prompt = outline._build_precheck_prompt(BEES, ["A title"])

    def test_four_kinds_are_offered(self):
        self.assertEqual(("causal", "explanatory", "comparative", "descriptive"), outline._TOPIC_SHAPES)
        for shape in outline._TOPIC_SHAPES:
            self.assertIn(f"  - {shape}:", self.prompt)
        self.assertIn('"causal" | "explanatory" | "comparative" | "descriptive"', self.prompt)

    def test_open_questions_have_examples_and_an_empty_cause(self):
        self.assertIn("Why are honeybee colonies declining?", self.prompt)
        self.assertIn('topic_cause="", topic_outcome="decline of honeybee colonies"', self.prompt)
        self.assertIn("never supply a cause yourself", self.prompt)

    def test_a_question_about_causes_is_no_longer_called_causal(self):
        self.assertNotIn("'what causes X?' (causal)", self.prompt)
        self.assertIn("'what causes Y?' (explanatory)", self.prompt)

    def test_corpus_fit_instructions_are_those_of_15_19_4(self):
        part = self.prompt[self.prompt.index("## PART 2"):self.prompt.index("## PART 3")]
        self.assertEqual("0d2e598546c44d7253cf", hashlib.sha256(part.encode("utf-8")).hexdigest()[:20])

    def test_prompt_version_changed(self):
        self.assertIn("15.19.5", outline._PRECHECK_PROMPT_VERSION)
        for version in (outline._CLUSTER_PROMPT_VERSION, outline._POSTURE_PROMPT_VERSION,
                        outline._ORDER_PROMPT_VERSION):
            self.assertIn("15.19.5", version)


class CauseGuardTests(unittest.TestCase):
    def check(self, topic, shape, cause, outcome, **kwargs):
        return outline._validate_precheck(stage0(shape, "PROCEED", cause, outcome), topic, **kwargs)

    def test_a_cause_the_topic_names_is_kept(self):
        cases = [
            ("Sleep deprivation impairs immune function.", "sleep deprivation", "impaired immune function"),
            ("Catalyst surface area determines reaction rate.", "catalyst surface area", "reaction rate"),
            (PRESS, "the printing press", "literacy in early modern Europe"),
            ("Does AI raise productivity?", "AI", "higher productivity"),
            ("Glacier surges follow changes in basal water pressure.", "basal water pressure changes", "glacier surges"),
            ("Sources of funding determine research output.", "sources of funding", "research output"),
            ("Market forces explain why vinyl records came back.", "market forces", "the return of vinyl records"),
        ]
        for topic, cause, outcome in cases:
            with self.subTest(topic=topic):
                result = self.check(topic, "causal", cause, outcome)
                self.assertEqual("causal", result["topic_shape"])
                self.assertEqual(cause, result["topic_cause"])
                self.assertNotIn("shape_adjusted", result)

    def test_an_empty_cause_makes_the_topic_explanatory(self):
        result = self.check(BEES, "causal", "", "decline of honeybee colonies")
        self.assertEqual("explanatory", result["topic_shape"])
        self.assertEqual("", result["topic_cause"])
        self.assertEqual("decline of honeybee colonies", result["topic_outcome"])
        self.assertEqual("causal_without_cause", result["shape_adjusted"])
        self.assertEqual("causal", result["model_topic_shape"])

    def test_a_placeholder_cause_is_refused(self):
        cases = [
            (BEES, "unknown factors"),
            (BEES, "unspecified drivers of the decline of honeybee colonies"),
            (BEES, "various causes"),
            (BEES, "the underlying determinants of colony decline"),
            ("Why do some glaciers surge?", "drivers of glacier surges"),
            ("How did the Antarctic ozone hole come about?", "the main reasons for the ozone hole"),
        ]
        for topic, cause in cases:
            with self.subTest(cause=cause):
                result = self.check(topic, "causal", cause, "the outcome named by the topic")
                self.assertEqual("explanatory", result["topic_shape"])
                self.assertEqual("", result["topic_cause"])
                self.assertEqual("cause_placeholder", result["shape_adjusted"])
                self.assertEqual(cause, result["model_topic_cause"])

    def test_a_cause_in_words_foreign_to_the_topic_is_refused(self):
        for cause in ("pesticide exposure", "the varroa mite and habitat loss", "climate change"):
            with self.subTest(cause=cause):
                result = self.check(BEES, "causal", cause, "decline of honeybee colonies")
                self.assertEqual("explanatory", result["topic_shape"])
                self.assertEqual("cause_not_in_topic", result["shape_adjusted"])

    def test_a_cause_that_rewords_the_outcome_is_refused(self):
        result = self.check(BEES, "causal", "declining honeybee colonies", "decline of honeybee colonies")
        self.assertEqual("explanatory", result["topic_shape"])
        self.assertEqual("cause_repeats_outcome", result["shape_adjusted"])

    def test_a_generic_noun_the_topic_itself_asks_about_does_not_pass_as_a_cause(self):
        topic = "What factors explain the decline of honeybee colonies?"
        for cause in ("factors", "factors behind the decline of honeybee colonies"):
            with self.subTest(cause=cause):
                result = self.check(topic, "causal", cause, "decline of honeybee colonies")
                self.assertEqual("explanatory", result["topic_shape"])

    def test_an_explanatory_answer_never_carries_a_cause(self):
        result = self.check(BEES, "explanatory", "pesticides", "decline of honeybee colonies")
        self.assertEqual("explanatory", result["topic_shape"])
        self.assertEqual("", result["topic_cause"])
        self.assertEqual("cause_dropped_from_open_question", result["shape_adjusted"])
        clean = self.check(BEES, "explanatory", "", "decline of honeybee colonies")
        self.assertNotIn("shape_adjusted", clean)

    def test_an_outcome_is_required_for_both_kinds(self):
        self.assertIsNone(self.check(BEES, "explanatory", "", ""))
        self.assertIsNone(self.check(PRESS, "causal", "the printing press", ""))

    def test_comparative_and_descriptive_answers_are_left_alone(self):
        for shape in ("comparative", "descriptive"):
            result = self.check("Is A larger than B?", shape, "", "")
            self.assertEqual(shape, result["topic_shape"])
            self.assertNotIn("shape_adjusted", result)

    def test_a_refusal_is_valid_whatever_the_slots_hold(self):
        result = outline._validate_precheck(stage0("causal", "REFUSE", "", ""), "asdf qwerty")
        self.assertEqual("REFUSE", result["corpus_fit"])
        self.assertEqual("causal", result["topic_shape"])

    def test_words_are_not_compared_across_languages(self):
        topic = "La privation de sommeil affaiblit la fonction immunitaire."
        kept = self.check(topic, "causal", "sleep deprivation", "impaired immune function", compare_words=False)
        self.assertEqual("causal", kept["topic_shape"])
        with mock.patch.dict(os.environ, {"RRR_TOPIC_LANG": "fr", "RRR_CORPUS_LANG": "en"}):
            self.assertFalse(outline._cause_words_comparable())
        with mock.patch.dict(os.environ, {"RRR_TOPIC_LANG": "pt", "RRR_CORPUS_LANG": "pt"}):
            self.assertTrue(outline._cause_words_comparable())

    def test_a_topic_in_a_script_without_word_breaks_is_not_judged(self):
        self.assertEqual("", outline._cause_check("原因", "为什么蜜蜂数量下降", "蜜蜂数量下降"))

    def test_unknown_shapes_and_fits_are_invalid(self):
        self.assertIsNone(outline._validate_precheck(stage0("open", "PROCEED", "", "x"), BEES))
        self.assertIsNone(outline._validate_precheck(stage0("causal", "MAYBE", "a", "b"), BEES))
        self.assertIsNone(outline._validate_precheck(["not", "a", "dict"], BEES))


class Stage0CallTests(StubbedStages):
    def precheck(self, fake, topic=BEES):
        diagnostics = {}
        result = self.run_with(fake, outline.precheck, topic, TITLES, diagnostics=diagnostics)
        return result, diagnostics

    def test_a_valid_first_answer_costs_one_call(self):
        fake = FakeModel([stage0("explanatory", outcome="decline of honeybee colonies")])
        result, diagnostics = self.precheck(fake)
        self.assertEqual(1, len(fake.prompts))
        self.assertEqual("explanatory", result["topic_shape"])
        self.assertEqual(outline._PRECHECK_PROMPT_VERSION, result["prompt_version"])
        self.assertEqual({}, diagnostics)

    def test_the_answer_that_used_to_end_the_run_is_now_an_open_question(self):
        fake = FakeModel([stage0("causal", cause="", outcome="decline of honeybee colonies")])
        result, _ = self.precheck(fake)
        self.assertEqual(1, len(fake.prompts))
        self.assertEqual("explanatory", result["topic_shape"])
        self.assertEqual("PROCEED", result["corpus_fit"])

    def test_an_unusable_answer_is_asked_for_once_more(self):
        fake = FakeModel(["I think this is causal.", stage0("explanatory", outcome="decline of honeybee colonies")])
        result, diagnostics = self.precheck(fake)
        self.assertEqual(2, len(fake.prompts))
        self.assertTrue(fake.prompts[1].startswith(fake.prompts[0]))
        self.assertIn("Reminder", fake.prompts[1])
        self.assertEqual("explanatory", result["topic_shape"])
        self.assertEqual({}, diagnostics)

    def test_two_failures_are_a_breakdown_with_its_kind(self):
        cases = [
            (["no json here", "still none"], "malformed_json"),
            ([stage0("open", outcome="x"), stage0("causal", fit="PERHAPS", cause="a", outcome="b")], "invalid_answer"),
            ([RuntimeError("connection refused"), RuntimeError("connection refused")], "model_call_error"),
            (["", ""], "empty_reply"),
            ([RuntimeError("timeout"), "not json"], "malformed_json"),
        ]
        for replies, kind in cases:
            with self.subTest(kind=kind, first=str(replies[0])[:20]):
                fake = FakeModel(replies)
                result, diagnostics = self.precheck(fake)
                self.assertIsNone(result)
                self.assertEqual(2, len(fake.prompts))
                self.assertEqual(kind, diagnostics["failure"])

    def test_no_titles_is_reported_without_a_call(self):
        fake = FakeModel([])
        diagnostics = {}
        result = self.run_with(fake, outline.precheck, BEES, [], diagnostics=diagnostics)
        self.assertIsNone(result)
        self.assertEqual("no_titles", diagnostics["failure"])
        self.assertEqual([], fake.prompts)

    def test_the_guard_runs_inside_the_call(self):
        fake = FakeModel([stage0("causal", cause="unspecified drivers of colony decline",
                                 outcome="decline of honeybee colonies")])
        result, _ = self.precheck(fake)
        self.assertEqual("explanatory", result["topic_shape"])
        self.assertEqual("cause_placeholder", result["shape_adjusted"])


class Stage1Tests(StubbedStages):
    def test_explanatory_topics_are_grouped_by_explanation(self):
        claims = [{"doc_id": d["doc_id"], "claim": d["claim"]} for d in SUMMARIES]
        prompt = outline._build_cluster_prompt(BEES, claims, topic_shape="explanatory",
                                               topic_outcome="decline of honeybee colonies")
        self.assertIn("TOPIC OUTCOME: decline of honeybee colonies", prompt)
        self.assertIn("one cluster per explanation", prompt)
        self.assertIn("names no cause", prompt)
        self.assertIn("measure, date or describe the outcome", prompt)
        self.assertIn("Papers on a neighbouring question", prompt)
        self.assertIn("address a genuinely different question", prompt)
        self.assertIn("do not add one of your own", prompt)
        for d in SUMMARIES:
            self.assertIn(f"[{d['doc_id']}]", prompt)

    def test_the_plan_keeps_each_cluster_s_explanation(self):
        fake = FakeModel([{"clusters": [
            {"cluster_id": "C1", "doc_ids": ["Abel_2010", "Baum_2012"], "shared_thread": "pesticides and colony health",
             "shared_cause": "neonicotinoid pesticide exposure"},
            {"cluster_id": "C2", "doc_ids": ["Dahl_2016"], "shared_thread": "measured colony losses", "shared_cause": ""},
        ], "unassigned_doc_ids": []}])
        plan = self.run_with(fake, outline.cluster_papers, BEES, SUMMARIES, topic_shape="explanatory",
                             topic_outcome="decline of honeybee colonies")
        self.assertEqual("neonicotinoid pesticide exposure", plan["clusters"][0]["shared_cause"])
        self.assertEqual("", plan["clusters"][1]["shared_cause"])
        self.assertEqual(["Cruz_2014"], plan["unassigned_doc_ids"])


class Stage2Tests(StubbedStages):
    CLUSTER = {
        "cluster_id": "C1", "doc_ids": ["Abel_2010", "Baum_2012"],
        "shared_thread": "pesticides and colony health",
        "shared_cause": "neonicotinoid pesticide exposure",
        "claims_by_doc": {d["doc_id"]: d["claim"] for d in SUMMARIES[:2]},
    }
    EVIDENCE = {"Abel_2010": ["p.4: Colonies exposed to neonicotinoid pesticides lost foragers."], "Baum_2012": []}

    def posture(self, answer, cluster=None):
        fake = FakeModel([answer, answer])
        result = self.run_with(fake, outline.posture_cluster, BEES, "explanatory", cluster or self.CLUSTER,
                               self.EVIDENCE, topic_cause="", topic_outcome="decline of honeybee colonies")
        return result, fake

    def answer(self, relation, cause_quote="", outcome_quote=""):
        return {"relation": relation, "elaboration": "Pesticide exposure weakened colonies.",
                "reasoning_trace": "The cluster names pesticides.", "lead_doc_id": "Abel_2010",
                "internal_disagreement": "", "cluster_cause_quote": cause_quote,
                "cluster_outcome_quote": outcome_quote}

    def test_the_relations_place_a_cluster_against_the_outcome(self):
        self.assertEqual({"explains_outcome", "documents_outcome", "disputes_outcome", "adjacent"},
                         outline._RELATIONS_BY_SHAPE["explanatory"])
        prompt = outline._build_posture_prompt(BEES, "explanatory", self.CLUSTER, self.EVIDENCE,
                                               topic_cause="", topic_outcome="decline of honeybee colonies")
        self.assertIn("TOPIC OUTCOME (explanandum, pinned): decline of honeybee colonies", prompt)
        self.assertIn("TOPIC CAUSE: none.", prompt)
        self.assertIn("CLUSTER'S EXPLANATION (from Stage 1): neonicotinoid pesticide exposure", prompt)
        self.assertIn("[adjacent, disputes_outcome, documents_outcome, explains_outcome]", prompt)
        self.assertNotIn("same_as_topic_cause", prompt)
        self.assertNotIn("TEST C-THROUGH", prompt)

    def test_an_explanation_found_in_the_cluster_s_own_words_is_kept(self):
        result, fake = self.posture(self.answer(
            "explains_outcome", "neonicotinoid pesticides weakened colonies", "weakened colonies"))
        self.assertEqual(1, len(fake.prompts))
        self.assertEqual("explains_outcome", result["relation"])
        self.assertNotIn("mechanism_grounding", result)

    def test_an_explanation_the_cluster_does_not_carry_falls_back_to_adjacent(self):
        result, _ = self.posture(self.answer(
            "explains_outcome", "habitat fragmentation starved the hives", "weakened colonies"))
        self.assertEqual("adjacent", result["relation"])
        self.assertEqual("downgraded_explains_to_adjacent", result["mechanism_grounding"])

    def test_an_explanation_without_its_quote_is_not_accepted(self):
        result, fake = self.posture(self.answer("explains_outcome", "", "weakened colonies"))
        self.assertIsNone(result)
        self.assertEqual(2, len(fake.prompts))

    def test_documenting_and_disputing_need_the_outcome_quote(self):
        for relation in ("documents_outcome", "disputes_outcome"):
            with self.subTest(relation=relation):
                kept, _ = self.posture(self.answer(relation, "", "Managed colonies fell by a quarter"))
                self.assertEqual(relation, kept["relation"])
                refused, _ = self.posture(self.answer(relation, "", ""))
                self.assertIsNone(refused)

    def test_relations_of_another_kind_are_not_accepted(self):
        for relation in ("rival_to_topic_cause", "confirms_description", "endorses_topic"):
            with self.subTest(relation=relation):
                result, _ = self.posture(self.answer(relation, "a cause quote here", "an outcome quote here"))
                self.assertIsNone(result)

    def test_explanatory_relations_are_not_accepted_for_a_causal_topic(self):
        self.assertIsNone(outline._validate_posture(
            self.answer("explains_outcome", "a cause quote here", "an outcome quote here"),
            "causal", {"Abel_2010"}))


class WholeOutlineTests(StubbedStages):
    def router(self, prompt):
        if "PRE-FLIGHT" in prompt:
            return reply(stage0("causal", cause="unknown factors", outcome="decline of honeybee colonies"))
        if "Group these papers" in prompt:
            return reply({"clusters": [
                {"cluster_id": "C1", "doc_ids": ["Abel_2010", "Baum_2012"],
                 "shared_thread": "pesticides and colony health", "shared_cause": "neonicotinoid pesticide exposure"},
                {"cluster_id": "C2", "doc_ids": ["Cruz_2014"],
                 "shared_thread": "parasites and viruses", "shared_cause": "varroa mite infestation"},
                {"cluster_id": "C3", "doc_ids": ["Dahl_2016"],
                 "shared_thread": "measured colony losses", "shared_cause": ""},
            ], "unassigned_doc_ids": []})
        if "CLUSTER LABEL (from Stage 1): pesticides" in prompt:
            return reply({"relation": "explains_outcome", "elaboration": "Pesticides weakened colonies.",
                          "reasoning_trace": "t", "lead_doc_id": "Abel_2010", "internal_disagreement": "",
                          "cluster_cause_quote": "neonicotinoid pesticides weakened colonies",
                          "cluster_outcome_quote": "weakened colonies"})
        if "CLUSTER LABEL (from Stage 1): parasites" in prompt:
            return reply({"relation": "explains_outcome", "elaboration": "The varroa mite killed colonies.",
                          "reasoning_trace": "t", "lead_doc_id": "Cruz_2014", "internal_disagreement": "",
                          "cluster_cause_quote": "the varroa mite and the viruses it carries killed colonies",
                          "cluster_outcome_quote": "killed colonies"})
        if "CLUSTER LABEL (from Stage 1): measured" in prompt:
            return reply({"relation": "documents_outcome", "elaboration": "Colonies fell by a quarter.",
                          "reasoning_trace": "t", "lead_doc_id": "Dahl_2016", "internal_disagreement": "",
                          "cluster_cause_quote": "",
                          "cluster_outcome_quote": "measures colony losses across forty countries"})
        if "ordered_cluster_ids" in prompt:
            return reply({"ordered_cluster_ids": ["C3", "C1", "C2"]})
        raise AssertionError("unexpected prompt: " + prompt[:80])

    def test_an_open_question_goes_through_every_stage(self):
        fake = FakeModel(router=self.router)
        plan = self.run_with(fake, outline.build_outline, BEES, SUMMARIES)
        self.assertEqual("explanatory", plan["topic_shape"])
        self.assertEqual("", plan["topic_cause"])
        self.assertEqual("decline of honeybee colonies", plan["topic_outcome"])
        self.assertEqual("cause_placeholder", plan["shape_adjusted"])
        self.assertEqual("unknown factors", plan["model_topic_cause"])
        self.assertEqual({"explains_outcome": 3, "documents_outcome": 1}, plan["relation_distribution"])
        self.assertEqual(["C3", "C1", "C2"], plan["ordered_cluster_ids"])
        self.assertEqual(["neonicotinoid pesticide exposure", "varroa mite infestation", ""],
                         [c["shared_cause"] for c in plan["clusters"]])
        self.assertEqual(1, sum("PRE-FLIGHT" in p for p in fake.prompts))

    def test_a_stage_0_answer_handed_in_is_not_asked_for_again(self):
        fake = FakeModel(router=self.router)
        early = outline._validate_precheck(stage0("explanatory", outcome="decline of honeybee colonies"), BEES)
        plan = self.run_with(fake, outline.build_outline, BEES, SUMMARIES, precheck_result=early)
        self.assertEqual("explanatory", plan["topic_shape"])
        self.assertEqual(0, sum("PRE-FLIGHT" in p for p in fake.prompts))
        self.assertNotIn("shape_adjusted", plan)

    def test_a_refusal_handed_in_stops_before_stage_1(self):
        fake = FakeModel(router=self.router)
        early = outline._validate_precheck(stage0("descriptive", fit="REFUSE"), "asdf qwerty")
        plan = self.run_with(fake, outline.build_outline, "asdf qwerty", SUMMARIES, precheck_result=early)
        self.assertTrue(plan["refused"])
        self.assertEqual("corpus_off_topic", plan["refusal_reason"])
        self.assertEqual([], fake.prompts)

    def test_a_failure_names_its_stage(self):
        def no_plan(prompt):
            if "PRE-FLIGHT" in prompt:
                return reply(stage0("explanatory", outcome="decline of honeybee colonies"))
            return {"message": {"content": "not json"}}

        diagnostics = {}
        plan = self.run_with(FakeModel(router=no_plan), outline.build_outline, BEES, SUMMARIES,
                             diagnostics=diagnostics)
        self.assertIsNone(plan)
        self.assertEqual("stage1", diagnostics["stage"])

        diagnostics = {}
        plan = self.run_with(FakeModel(router=lambda p: {"message": {"content": "not json"}}),
                             outline.build_outline, BEES, SUMMARIES, diagnostics=diagnostics)
        self.assertIsNone(plan)
        self.assertEqual("stage0", diagnostics["stage"])
        self.assertEqual("malformed_json", diagnostics["failure"])

    def test_a_claim_still_takes_the_causal_path(self):
        def router(prompt):
            if "PRE-FLIGHT" in prompt:
                return reply(stage0("causal", cause="the printing press", outcome="higher literacy"))
            if "Group these papers" in prompt:
                self.assertIn("TOPIC SHAPE: causal", prompt)
                self.assertNotIn("EXPLANATION", prompt)
                return reply({"clusters": [{"cluster_id": "C1", "doc_ids": [d["doc_id"] for d in SUMMARIES],
                                            "shared_thread": "print and reading", "shared_cause": "cheap printed books"}],
                              "unassigned_doc_ids": []})
            self.assertIn("TOPIC CAUSE (explanans, pinned): the printing press", prompt)
            return reply({"relation": "same_as_topic_cause", "elaboration": "Print raised literacy.",
                          "reasoning_trace": "cheap printed books raised literacy", "lead_doc_id": "Abel_2010",
                          "internal_disagreement": "", "cluster_cause_quote": "cheap printed books spread",
                          "cluster_outcome_quote": "literacy rose in towns"})

        plan = self.run_with(FakeModel(router=router), outline.build_outline, PRESS, SUMMARIES)
        self.assertEqual("causal", plan["topic_shape"])
        self.assertEqual("the printing press", plan["topic_cause"])
        self.assertEqual({"same_as_topic_cause": 4}, plan["relation_distribution"])


if __name__ == "__main__":
    unittest.main()
