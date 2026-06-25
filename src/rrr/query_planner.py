import os, json, re, time


_PLANNER_OPTIONS = {
    "temperature": float(os.environ.get("RRR_PLANNER_T", "0.1")),
    "num_ctx": int(os.environ.get("RRR_PLANNER_CTX", "2048")),
    "num_predict": int(os.environ.get("RRR_PLANNER_PRED", "700")),
}


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
        print(f"[Planner] mode=llm n_must={len(obj['keywords_must'])} n_any={len(obj['keywords_any'])} n_probes={len(obj['probes'])}")
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
