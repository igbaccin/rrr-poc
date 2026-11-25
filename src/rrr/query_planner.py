import os, json, re, time


_PLANNER_OPTIONS = {"temperature": 0.0, "num_ctx": 2048, "num_predict": 700}
_TERMS_OPTIONS = {"temperature": 0.0, "num_ctx": 2048, "num_predict": 300}
_PROBE_CAP = 12


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


def _ensure_probes(topic: str, obj: dict, insert_raw_topic: bool = True):
    probes = _clean_list(obj.get("probes", []), 8, item_limit=120)
    if not probes:
        must = obj.get("keywords_must", [])
        any_terms = obj.get("keywords_any", [])
        if must:
            probes.append(" ".join(must[:6]))
        if any_terms:
            probes.append(" ".join(any_terms[:8]))
    if insert_raw_topic and topic and topic.lower() not in probes:
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

def _extract_terms_of_art(topic: str, plan_obj: dict, model: str, metrics=None,
                          corpus_lang: str = "en", topic_lang: str = "en"):
    start = time.perf_counter()
    try:
        import ollama
        from rrr.language import language_name

        existing_terms = ", ".join(
            (plan_obj.get("keywords_must", []) or []) +
            (plan_obj.get("keywords_any", []) or [])
        )
        cross_lang = bool(corpus_lang) and bool(topic_lang) and corpus_lang != topic_lang
        corpus_lang_name = language_name(corpus_lang)
        xlang_instr = (
            f"The topic is in {language_name(topic_lang)} but the corpus is "
            f"in {corpus_lang_name}. Emit all terms of art IN "
            f"{corpus_lang_name.upper()}.\n"
            if cross_lang else ""
        )
        prompt = (
            "You are decomposing a scholarly research topic into the vocabulary "
            "that source pages would actually use.\n\n"
            + xlang_instr +
            "Topic: " + topic + "\n"
            "Topic words already covered by stage 1: " + (existing_terms or "(none)") + "\n\n"
            f"List 4-8 TECHNICAL TERMS OF ART (in {corpus_lang_name}) from the "
            "relevant academic subfield that would appear in source pages but "
            "are NOT in the topic statement.\n"
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


_TOPIC_REFORM_OPTIONS = {"temperature": 0.0, "num_ctx": 2048, "num_predict": 400}


def _reformulate_topic(topic: str, model: str, metrics=None):
    start = time.perf_counter()
    try:
        import ollama
        import os
        from rrr.language import language_directive
        topic_lang = os.environ.get("RRR_TOPIC_LANG", "en")
        _lang_directive = language_directive(topic_lang)
        _lang_pfx = f"{_lang_directive}\n\n" if _lang_directive else ""
        _same_lang_clause = (
            "3. CRITICAL: the topic_question and every topic_dimensions "
            "entry MUST be written in the SAME LANGUAGE as the user's "
            "topic above. Do NOT translate to English. If the user wrote "
            "French, respond in French; Spanish → Spanish; Chinese → "
            "Chinese; etc.\n\n"
            if topic_lang != "en" else ""
        )
        prompt = (
            _lang_pfx +
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
            "measurement, period, region).\n"
            + _same_lang_clause +
            "\n"
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
    out = list(probes)
    seen_lower = {p.lower().strip() for p in out if p}
    for t in terms or []:
        t_clean = (t or "").strip().lower()
        if not t_clean or t_clean in seen_lower:
            continue
        if any(t_clean in p.lower() or p.lower() in t_clean for p in out if p):
            continue
        out.append(t_clean)
        seen_lower.add(t_clean)
        if cap is not None and len(out) >= cap:
            break
    return out


def plan(topic: str, metrics=None, corpus_lang: str = "en", topic_lang: str = "en"):
    model = os.environ.get("RRR_PLANNER_MODEL", os.environ.get("RRR_MODEL", "mistral-small:24b"))
    start = time.perf_counter()
    cross_lang = bool(corpus_lang) and bool(topic_lang) and corpus_lang != topic_lang
    try:
        import ollama
        from rrr.language import language_name
        corpus_lang_name = language_name(corpus_lang)

        if cross_lang:
            topic_lang_name = language_name(topic_lang)
            xlang_instr = (
                f"IMPORTANT: the topic is written in {topic_lang_name}, but "
                f"the document corpus you will search is written in "
                f"{corpus_lang_name}. Emit EVERY keyword, token, and probe "
                f"IN {corpus_lang_name.upper()} — translate the topic's "
                f"concepts into the scholarly vocabulary a {corpus_lang_name} "
                f"paper would actually use. Do NOT emit {topic_lang_name} "
                f"terms; they would not match the {corpus_lang_name} corpus.\n\n"
            )
        else:
            xlang_instr = ""

        prompt = (
            "Extract search terms for a scholarly retrieval plan.\n"
            + xlang_instr +
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
        from rrr.utils import extract_first_json
        obj = extract_first_json(raw)
        if obj is None:
            raise ValueError("no JSON object in planner response")

        for k in ("keywords_must", "keywords_any", "exclude", "probes"):
            if k not in obj or not isinstance(obj[k], list):
                obj[k] = []

        obj["keywords_must"] = _clean_list(obj["keywords_must"], 8, item_limit=60)
        obj["keywords_any"]  = _clean_list(obj["keywords_any"], 12, item_limit=60)
        obj["exclude"]       = _clean_list(obj["exclude"], 8, item_limit=60)
        obj = _ensure_probes(topic, obj, insert_raw_topic=not cross_lang)
        duration_s = time.perf_counter() - start
        obj["planner_meta"] = {
            "mode": "llm",
            "model": model,
            "duration_s": round(duration_s, 4),
            "corpus_lang": corpus_lang,
            "topic_lang": topic_lang,
            "cross_lang_retrieval": cross_lang,
        }

        terms = _extract_terms_of_art(topic, obj, model, metrics=metrics,
                                      corpus_lang=corpus_lang, topic_lang=topic_lang)
        if terms:
            merged = _merge_probes_with_terms(obj["probes"], terms, cap=_PROBE_CAP)
            added = len(merged) - len(obj["probes"])
            obj["probes"] = merged
            obj["terms_of_art"] = terms
            obj["planner_meta"]["mode"] = "llm_two_stage"
            obj["planner_meta"]["terms_of_art_count"] = len(terms)
            obj["planner_meta"]["probes_added_by_terms"] = added
            print(f"[Planner] stage2 terms_of_art={len(terms)} probes_added={added} total_probes={len(obj['probes'])}")

        obj["topic_display"] = topic
        obj["topic_question"] = topic
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
        obj = _ensure_probes(topic, _heuristic_plan(topic),
                             insert_raw_topic=not cross_lang)
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
