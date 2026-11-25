import hashlib
import json
import os
import sys
import unittest
from unittest import mock

from rrr import outline
import rrr.writer as writer


HOSTED = hasattr(writer, "_WRITER_HOST_EXECUTION_PROFILE")

TOPIC = "Catalyst surface area determines reaction rate."
DOC_CLAIMS = [
    {"doc_id": "Adams_2001", "claim": "The paper argues that surface area governs the rate."},
    {"doc_id": "Baker_2005", "claim": "The paper argues that temperature dominates, contra surface-area accounts."},
    {"doc_id": "Chen_2010", "claim": ""},
]
CLUSTER = {
    "cluster_id": "C1",
    "doc_ids": ["Adams_2001", "Baker_2005"],
    "shared_thread": "surface area and reaction kinetics",
    "shared_cause": "greater catalyst surface area",
    "claims_by_doc": {
        "Adams_2001": "The paper argues that surface area governs the rate.",
        "Baker_2005": "The paper argues that temperature dominates, contra surface-area accounts.",
    },
}
EVIDENCE = {"Adams_2001": ["p.3: Rates rose with surface area."], "Baker_2005": []}
ORDER_SUMMARIES = [
    {"cluster_id": "C1", "relation": "adjacent", "elaboration": "Rates rise with area.",
     "shared_thread": "surface area and reaction kinetics"},
    {"cluster_id": "C2", "relation": "adjacent", "elaboration": "Heat sets the pace.",
     "shared_thread": "temperature and activation"},
    {"cluster_id": "C3", "relation": "adjacent", "elaboration": "Poisoning slows catalysts.",
     "shared_thread": "catalyst poisoning"},
]
OLD_SHAPES = ("causal", "comparative", "descriptive")
OLD_RELATIONS = (
    "same_as_topic_cause", "upstream_of_topic_cause", "downstream_of_topic_cause",
    "rival_to_topic_cause", "scope_condition", "adjacent",
    "endorses_topic", "reverses_topic", "qualifies_topic", "methodological_critique",
    "confirms_description", "contradicts_description", "adds_nuance", "",
)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:20]


def _clean_env():
    env = dict(os.environ)
    for name in ("RRR_CORPUS_LANG", "RRR_TOPIC_LANG", "RRR_WRITER_PREV_TAIL_STRONG"):
        env.pop(name, None)
    return env


def outline_fingerprints() -> dict:
    out = {}
    with mock.patch.dict(os.environ, _clean_env(), clear=True):
        for shape in OLD_SHAPES:
            outcome = "faster reaction rate" if shape == "causal" else ""
            cause = "catalyst surface area" if shape == "causal" else ""
            out[f"cluster:{shape}"] = _sha(outline._build_cluster_prompt(
                TOPIC, DOC_CLAIMS, topic_shape=shape, topic_outcome=outcome))
            out[f"posture:{shape}"] = _sha(outline._build_posture_prompt(
                TOPIC, shape, CLUSTER, EVIDENCE, topic_cause=cause, topic_outcome=outcome))
        out["order"] = _sha(outline._build_order_prompt(TOPIC, ORDER_SUMMARIES))
    return out


def writer_fingerprints() -> dict:
    out = {}
    with mock.patch.dict(os.environ, _clean_env(), clear=True):
        for relation in OLD_RELATIONS:
            kwargs = {"outline_block": "BRIEF", "claims_so_far": "", "relation": relation,
                      "topic_shape": "causal"}
            if not HOSTED:
                kwargs.update(lead_surname="Adams", forbidden_opener_surnames=["Baker"])
            out[f"stream:{relation or 'none'}"] = _sha(writer._build_stream_prompt(
                TOPIC, "surface area and reaction kinetics", "EVIDENCE", "ALLOWED",
                "The previous section ended here.", **kwargs))
        out["opening"] = _sha(writer._build_opening_prompt(TOPIC, "SUMMARY", "EVIDENCE", "ALLOWED"))
        out["openers"] = _sha(json.dumps(
            {k: writer._RELATION_OPENERS[k] for k in OLD_RELATIONS if k}, sort_keys=True))
    return out


PINS = {
    "outline": {
        "cluster:causal": "02d01c59dd288e39103a",
        "posture:causal": "fb9d7a83a9cbb9862fb4",
        "cluster:comparative": "1bfddb746661a1e3736a",
        "posture:comparative": "1e7124910309824bdbdd",
        "cluster:descriptive": "572e1c1a1a6f33d753fa",
        "posture:descriptive": "c046fe84c39e00eca6a3",
        "order": "c6ee69b65354e0fce860",
    },
    "local": {
        "stream:same_as_topic_cause": "1fcb561e077cc700a0c0",
        "stream:upstream_of_topic_cause": "40cf426ac6a0bf7a52a6",
        "stream:downstream_of_topic_cause": "f7983b92b9ecd87ed98d",
        "stream:rival_to_topic_cause": "8a5b68fdc308d508cc9d",
        "stream:scope_condition": "96e6ad099a91eff18499",
        "stream:adjacent": "544eb905d57ab79eb574",
        "stream:endorses_topic": "acb9e178fa2658f8d7db",
        "stream:reverses_topic": "abbf445f5a03df39d4fb",
        "stream:qualifies_topic": "47060d149521f4a7c038",
        "stream:methodological_critique": "aea623d79c4a67227d8c",
        "stream:confirms_description": "3005db8f60592b64d58a",
        "stream:contradicts_description": "d692201013773cc2d183",
        "stream:adds_nuance": "0860199153517eae84c5",
        "stream:none": "e9c53a5df42b7cbc77d9",
        "opening": "4a9da053666754bc2eaa",
        "openers": "3f6bfccac0b83a6461c4",
    },
    "hosted": {
        "stream:same_as_topic_cause": "43b4ba0bd17ca4f2fe76",
        "stream:upstream_of_topic_cause": "a557c1359f4940364a79",
        "stream:downstream_of_topic_cause": "cabc7e53cc33ad161562",
        "stream:rival_to_topic_cause": "410578c07623601ebf2e",
        "stream:scope_condition": "79fc31516024b17b5eae",
        "stream:adjacent": "b013d413efea10c3e05c",
        "stream:endorses_topic": "fb967feed21af9d1e659",
        "stream:reverses_topic": "e03dccb5a642e681a335",
        "stream:qualifies_topic": "12789f2e2450caaebfb9",
        "stream:methodological_critique": "4d2f095de92359d7609f",
        "stream:confirms_description": "254a71a171565c1bd166",
        "stream:contradicts_description": "f4d8711a416e7e655dd4",
        "stream:adds_nuance": "ef37507c8e289c235a50",
        "stream:none": "107a627b32c7ff777807",
        "opening": "e8ad4d54c09dc3c80e82",
        "openers": "3f6bfccac0b83a6461c4",
    },
}


class PromptPinTests(unittest.TestCase):
    maxDiff = None

    def test_stage_one_to_three_instructions_are_unchanged_for_the_three_older_kinds(self):
        self.assertEqual(PINS["outline"], outline_fingerprints())

    def test_writer_instructions_are_unchanged_for_the_older_relations(self):
        self.assertEqual(PINS["hosted" if HOSTED else "local"], writer_fingerprints())


if __name__ == "__main__":
    if "--print" in sys.argv:
        print(json.dumps({"outline": outline_fingerprints(),
                          "hosted" if HOSTED else "local": writer_fingerprints()}, indent=4))
    else:
        unittest.main()
