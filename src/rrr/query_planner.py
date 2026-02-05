import os, json

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
        "exclude": []
    }

def plan(topic: str):
    try:
        import ollama

        prompt = (
            "Extract search terms for a scholarly retrieval plan.\n"
            "Topic: " + topic + "\n\n"
            "Return ONLY a JSON object with keys: keywords_must, keywords_any, exclude.\n"
            "Each value must be an array of short lowercase tokens (1-3 words each).\n"
        )

        res = ollama.chat(
            model=os.environ.get("RRR_PLANNER_MODEL", os.environ.get("RRR_MODEL", "mistral")),
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.1, "num_ctx": 1024},
            keep_alive="5m",
            stream=False,
        )
        raw = res["message"]["content"].strip()
        obj = json.loads(raw[raw.find("{"):raw.rfind("}") + 1])

        for k in ("keywords_must", "keywords_any", "exclude"):
            if k not in obj or not isinstance(obj[k], list):
                obj[k] = []

        obj["keywords_must"] = [x.lower()[:60] for x in obj["keywords_must"]][:8]
        obj["keywords_any"]  = [x.lower()[:60] for x in obj["keywords_any"]][:12]
        obj["exclude"]       = [x.lower()[:60] for x in obj["exclude"]][:8]
        return obj
    except Exception:
        return _heuristic_plan(topic)
