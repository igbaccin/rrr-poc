import os, json, re, time


_PLANNER_OPTIONS = {
    "temperature": float(os.environ.get("RRR_PLANNER_T", "0.1")),
    "num_ctx": int(os.environ.get("RRR_PLANNER_CTX", "2048")),
    "num_predict": int(os.environ.get("RRR_PLANNER_PRED", "700")),
}

# v9 (R7): second-stage planner asks for domain technical vocabulary the topic
# statement does not contain. Smaller budget than stage 1 because we want a
# tight list of phrases.
_TERMS_OPTIONS = {
    "temperature": float(os.environ.get("RRR_PLANNER_TERMS_T", "0.1")),
    "num_ctx": int(os.environ.get("RRR_PLANNER_TERMS_CTX", "2048")),
    "num_predict": int(os.environ.get("RRR_PLANNER_TERMS_PRED", "300")),
}

# v9 (R7): max probes after merging stage 1 + stage 2. Stage 1 caps at 8; stage
# 2 can add up to (PROBE_CAP - len(stage1)) novel technical phrases.
_PROBE_CAP = int(os.environ.get("RRR_PLANNER_PROBE_CAP", "12"))


def _clean_list(values, limit, item_limit=80):
    out = []
    if not isinstance(values, list):
        return out
    for value in values:
        text = re.sub(r"\s+", " ", str(value or "").strip().lower())
        if text and text not in out:
            out.append(text[:item_limit])
        if len(out) >= limit:
            break
    return out


def _ensure_probes(topic: str, obj: dict):
    probes = _clean_list(obj.get("probes", []), 8, item_limit=120)
    if not probes:
        must = obj.get("keywords_must", [])
        any_terms = obj.get("keywords_any", [])
        if must:
            probes.append(" ".join(must[:6]))
        if any_terms:
            probes.append(" ".join(any_terms[:8]))
    if topic and topic.lower() not in probes:
        probes.insert(0, topic.lower())
    obj["probes"] = probes[:8]
    return obj

def _heuristic_plan(topic: str):
    toks = [t.strip(",.;:()[]").lower() for t in topic.split()]
    toks = [t for t in toks if len(t) >= 4]

    uniq = []
    for t in toks:
        if t not in uniq:
            uniq.append(t)

    must = uniq[:6]
    any_terms = uniq[6:18]

    return {
        "keywords_must": must,
        "keywords_any": any_terms,
        "exclude": [],
        "probes": [topic.lower(), " ".join(must + any_terms[:4]).strip()]
    }

def _extract_terms_of_art(topic: str, plan_obj: dict, model: str, metrics=None):
    """v9 (R7): second-stage planner call that asks for technical vocabulary
    the topic statement does not contain. Returns a list of short phrases, or
    [] on any failure (this stage is purely additive — a failure means we keep
    stage-1 probes only, not a degraded plan).
    """
    start = time.perf_counter()
    try:
        import ollama

        existing_terms = ", ".join(
            (plan_obj.get("keywords_must", []) or []) +
            (plan_obj.get("keywords_any", []) or [])
        )
        prompt = (
            "You are decomposing a scholarly research topic into the vocabulary "
            "that source pages would actually use.\n\n"
            "Topic: " + topic + "\n"
            "Topic words already covered by stage 1: " + (existing_terms or "(none)") + "\n\n"
            "List 4-8 TECHNICAL TERMS OF ART from the relevant academic subfield "
            "that would appear in source pages but are NOT in the topic statement.\n"
            "Examples (for 'institutions are the fundamental cause of long-run "
            "economic growth'): property rights, contract enforcement, "
            "settler mortality, factor endowments, credible commitment, "
            "extractive institutions.\n"
            "Each phrase should be 2-5 lowercase words. Avoid generic words "
            "already in the topic.\n\n"
            "Return ONLY a JSON object with one key: terms_of_art (array of "
            "phrases)."
        )
        res = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options=_TERMS_OPTIONS,
            keep_alive="5m",
            format="json",
            stream=False,
        )
        raw = (res.get("message", {}).get("content") or "").strip()
        # tolerate either a clean JSON or a JSON-with-prefix
        start_idx = raw.find("{")
        end_idx = raw.rfind("}")
        if start_idx < 0 or end_idx <= start_idx:
            return []
        obj = json.loads(raw[start_idx:end_idx + 1])
        terms = _clean_list(obj.get("terms_of_art", []), 8, item_limit=80)
        duration_s = time.perf_counter() - start
        if metrics:
            metrics.record_llm(
                "planner_terms_of_art", model, options=_TERMS_OPTIONS,
                duration_s=duration_s, prompt_chars=len(prompt),
                response_chars=len(raw),
            )
        return terms
    except Exception as e:
        if metrics:
            metrics.record_llm(
                "planner_terms_of_art", model, options=_TERMS_OPTIONS,
                success=False, duration_s=time.perf_counter() - start, error=e,
            )
        return []


# v12: topic reformulation. Normalises any user-typed topic into a
# scholarly-question form for internal use, while leaving the user's original
# string on the rendered output's title. The pipeline currently threads the
# raw topic into every writer prompt, which produces meta-commentary and
# defend/attack framing when the user writes a strong thesis. Reformulating
# upstream eliminates that pressure.
_TOPIC_REFORM_OPTIONS = {
    "temperature": float(os.environ.get("RRR_TOPIC_REFORM_T", "0.0")),
    "num_ctx": int(os.environ.get("RRR_TOPIC_REFORM_CTX", "2048")),
    "num_predict": int(os.environ.get("RRR_TOPIC_REFORM_PRED", "400")),
}


def _reformulate_topic(topic: str, model: str, metrics=None):
    """v12: turn the user's raw topic into a scholarly investigation question.

    Returns a dict {topic_question, topic_dimensions} or None on any failure
    (caller falls back to using the raw topic for both surfaces).
    """
    start = time.perf_counter()
    try:
        import ollama
        prompt = (
            "Read the user's research topic. Produce a normalised internal "
            "version that the literature-review writer will use as its frame.\n\n"
            "Goals:\n"
            "1. Convert any thesis or assertion into a scholarly investigation "
            "question. The writer must investigate, not defend or attack.\n"
            "   - 'X causes Y' becomes 'What is the role of X in Y?'\n"
            "   - 'Did Z happen?' stays as is.\n"
            "   - A bare phrase like 'X and Y' becomes 'What is the role of X "
            "in shaping Y?' (fill in the implicit verb).\n"
            "2. Name 3-5 dimensions of disagreement the literature would have "
            "around this question (e.g. mechanism, scope conditions, "
            "measurement, period, region).\n\n"
            "Topic from user:\n" + topic + "\n\n"
            "Return ONLY a JSON object with two keys:\n"
            "  topic_question: a single sentence ending with '?'\n"
            "  topic_dimensions: array of 3-5 short phrases (2-5 words each)"
        )
        res = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options=_TOPIC_REFORM_OPTIONS,
            keep_alive="5m",
            format="json",
            stream=False,
        )
        raw = (res.get("message", {}).get("content") or "").strip()
        start_idx = raw.find("{")
        end_idx = raw.rfind("}")
        if start_idx < 0 or end_idx <= start_idx:
            return None
        obj = json.loads(raw[start_idx:end_idx + 1])
        question = str(obj.get("topic_question", "")).strip()
        if not question:
            return None
        if not question.endswith("?"):
            question = question.rstrip(".") + "?"
        dimensions = obj.get("topic_dimensions", [])
        if not isinstance(dimensions, list):
            dimensions = []
        dimensions = [str(d).strip() for d in dimensions if str(d).strip()][:5]
        duration_s = time.perf_counter() - start
        if metrics:
            metrics.record_llm(
                "topic_reformulation", model, options=_TOPIC_REFORM_OPTIONS,
                duration_s=duration_s, prompt_chars=len(prompt),
                response_chars=len(raw),
            )
        return {"topic_question": question[:300], "topic_dimensions": dimensions}
    except Exception as e:
        if metrics:
            metrics.record_llm(
                "topic_reformulation", model, options=_TOPIC_REFORM_OPTIONS,
                success=False, duration_s=time.perf_counter() - start, error=e,
            )
        return None


def _merge_probes_with_terms(probes, terms, cap=None):
    """v9 (R7): merge stage-1 probes and stage-2 technical terms; reject
    near-duplicates via substring containment (avoids adding the rapidfuzz
    dependency to the planner module just for this).
    """
    out = list(probes)
    seen_lower = {p.lower().strip() for p in out if p}
    for t in terms or []:
        t_clean = (t or "").strip().lower()
        if not t_clean or t_clean in seen_lower:
            continue
        # reject if substring of an existing probe, or an existing probe is
        # a substring of it
        if any(t_clean in p.lower() or p.lower() in t_clean for p in out if p):
            continue
        out.append(t_clean)
        seen_lower.add(t_clean)
        if cap is not None and len(out) >= cap:
            break
    return out


def plan(topic: str, metrics=None):
    model = os.environ.get("RRR_PLANNER_MODEL", os.environ.get("RRR_MODEL", "mistral"))
    start = time.perf_counter()
    try:
        import ollama

        prompt = (
            "Extract search terms for a scholarly retrieval plan.\n"
            "Topic: " + topic + "\n\n"
            "Return ONLY a JSON object with keys: keywords_must, keywords_any, exclude, probes.\n"
            "keywords_must, keywords_any, and exclude must be arrays of short lowercase tokens.\n"
            "probes must be an array of 3 to 6 short search phrases that cover distinct subclaims.\n"
        )

        res = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options=_PLANNER_OPTIONS,
            keep_alive="5m",
            stream=False,
        )
        raw = res["message"]["content"].strip()
        obj = json.loads(raw[raw.find("{"):raw.rfind("}") + 1])

        for k in ("keywords_must", "keywords_any", "exclude", "probes"):
            if k not in obj or not isinstance(obj[k], list):
                obj[k] = []

        obj["keywords_must"] = _clean_list(obj["keywords_must"], 8, item_limit=60)
        obj["keywords_any"]  = _clean_list(obj["keywords_any"], 12, item_limit=60)
        obj["exclude"]       = _clean_list(obj["exclude"], 8, item_limit=60)
        obj = _ensure_probes(topic, obj)
        duration_s = time.perf_counter() - start
        obj["planner_meta"] = {
            "mode": "llm",
            "model": model,
            "duration_s": round(duration_s, 4),
        }

        # v9 (R7): second-stage technical-vocabulary call. Purely additive —
        # failure leaves stage-1 probes intact.
        if os.environ.get("RRR_PLANNER_TWO_STAGE", "1") != "0":
            terms = _extract_terms_of_art(topic, obj, model, metrics=metrics)
            if terms:
                merged = _merge_probes_with_terms(obj["probes"], terms, cap=_PROBE_CAP)
                added = len(merged) - len(obj["probes"])
                obj["probes"] = merged
                obj["terms_of_art"] = terms
                obj["planner_meta"]["mode"] = "llm_two_stage"
                obj["planner_meta"]["terms_of_art_count"] = len(terms)
                obj["planner_meta"]["probes_added_by_terms"] = added
                print(f"[Planner] stage2 terms_of_art={len(terms)} probes_added={added} total_probes={len(obj['probes'])}")

        # v12: topic reformulation. Falls back to raw topic on failure so the
        # rest of the pipeline never breaks on a None topic_question.
        obj["topic_display"] = topic
        obj["topic_question"] = topic
        if os.environ.get("RRR_TOPIC_REFORMULATION", "1") != "0":
            reform = _reformulate_topic(topic, model, metrics=metrics)
            if reform:
                obj["topic_question"] = reform["topic_question"]
                obj["topic_dimensions"] = reform["topic_dimensions"]
                obj["planner_meta"]["topic_reformulated"] = True
                print(f"[Planner] topic_question='{reform['topic_question']}'")
                if reform["topic_dimensions"]:
                    print(f"[Planner] topic_dimensions={reform['topic_dimensions']}")

        print(f"[Planner] mode={obj['planner_meta']['mode']} n_must={len(obj['keywords_must'])} n_any={len(obj['keywords_any'])} n_probes={len(obj['probes'])}")
        if metrics:
            metrics.record_llm("planner", model, options=_PLANNER_OPTIONS,
                               duration_s=duration_s, prompt_chars=len(prompt),
                               response_chars=len(raw))
        return obj
    except Exception as e:
        obj = _ensure_probes(topic, _heuristic_plan(topic))
        obj["planner_meta"] = {
            "mode": "heuristic_fallback",
            "model": model,
            "reason": str(e)[:300],
            "duration_s": round(time.perf_counter() - start, 4),
        }
        print(f"[Planner] mode=heuristic_fallback reason={str(e)[:120]} n_must={len(obj['keywords_must'])} n_any={len(obj['keywords_any'])} n_probes={len(obj['probes'])}")
        if metrics:
            metrics.record_llm("planner", model, options=_PLANNER_OPTIONS,
                               success=False, duration_s=time.perf_counter() - start,
                               error=e)
        return obj
