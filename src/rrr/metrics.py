from collections import defaultdict
from contextlib import contextmanager
import datetime as _dt
import json
import os
import re
import threading
import time

from rrr.paths import runs_path
from rrr.utils import ensure_dir


# v15.14: never persist credential-looking env values into run artifacts —
# run_metrics.json gets tarred, downloaded, and shared in replication
# bundles, and the API runtime makes RRR_*-prefixed credentials plausible.
_SENSITIVE_ENV_RE = re.compile(r"KEY|TOKEN|SECRET|PASS|CREDENTIAL", re.IGNORECASE)


def _redact_env_value(key: str, value):
    if value and _SENSITIVE_ENV_RE.search(key or ""):
        return "***redacted***"
    return value


class RunMetrics:
    def __init__(self, task: str, topic: str):
        self.task = task
        self.topic = topic
        self.started_at = _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"
        self._t0 = time.perf_counter()
        self._lock = threading.Lock()
        self.stages = defaultdict(lambda: {"calls": 0, "seconds": 0.0})
        self.counters = defaultdict(int)
        self.values = {}
        self.llm_calls = []
        self.cache = defaultdict(lambda: {"hits": 0, "misses": 0, "writes": 0, "skips": 0})

    @contextmanager
    def stage(self, name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.add_stage_time(name, time.perf_counter() - start)

    def add_stage_time(self, name: str, seconds: float):
        with self._lock:
            entry = self.stages[name]
            entry["calls"] += 1
            entry["seconds"] = round(entry["seconds"] + seconds, 4)

    def inc(self, key: str, amount: int = 1):
        with self._lock:
            self.counters[key] += amount

    def set(self, key: str, value):
        with self._lock:
            self.values[key] = value

    def cache_event(self, name: str, event: str, amount: int = 1):
        if event not in {"hits", "misses", "writes", "skips"}:
            event = "skips"
        with self._lock:
            self.cache[name][event] += amount

    def record_llm(self, stage: str, model: str, options=None, success: bool = True,
                   duration_s=None, prompt_chars=None, response_chars=None, error=None):
        call = {
            "stage": stage,
            "model": model,
            "success": bool(success),
        }
        if options:
            call["options"] = dict(options)
        if duration_s is not None:
            call["duration_s"] = round(float(duration_s), 4)
        if prompt_chars is not None:
            call["prompt_chars"] = int(prompt_chars)
        if response_chars is not None:
            call["response_chars"] = int(response_chars)
        if error:
            call["error"] = str(error)[:300]
        with self._lock:
            self.llm_calls.append(call)
            self.counters[f"llm_calls_{stage}"] += 1
            self.counters["llm_calls_total"] += 1
            if not success:
                self.counters[f"llm_failures_{stage}"] += 1
                self.counters["llm_failures_total"] += 1

    def to_dict(self):
        finished_at = _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"
        llm_by_stage = defaultdict(int)
        for call in self.llm_calls:
            llm_by_stage[call.get("stage", "unknown")] += 1
        return {
            "task": self.task,
            "topic": self.topic,
            "started_at": self.started_at,
            "finished_at": finished_at,
            "duration_s": round(time.perf_counter() - self._t0, 4),
            "stages": dict(self.stages),
            "counters": dict(self.counters),
            "values": dict(self.values),
            "cache": {k: dict(v) for k, v in self.cache.items()},
            "llm_calls_by_stage": dict(llm_by_stage),
            "llm_calls": list(self.llm_calls),
            "env": {
                k: _redact_env_value(k, os.environ.get(k))
                for k in sorted(os.environ)
                if k.startswith("RRR_") or k in {"OLLAMA_HOST", "OLLAMA_NUM_PARALLEL", "OLLAMA_MAX_LOADED_MODELS"}
            },
        }

    def save(self, path=None):
        # v15.14: atomic write — a crash mid-save left truncated metrics.
        out_path = str(path or runs_path("run_metrics.json"))
        ensure_dir(os.path.dirname(out_path))
        tmp = out_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        os.replace(tmp, out_path)
        return out_path
