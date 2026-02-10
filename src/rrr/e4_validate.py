"""
E4 Validation: Stance-Context Mismatch Detection

Checks whether papers tagged as CRITIQUES are cited in supporting context
without appropriate framing (and vice versa).
"""
import re
from typing import Dict, List, Tuple

CRITIQUE_PATTERNS = [
    r'\b(challenges?|challenged|challenging)\b',
    r'\b(critiques?|critiqued|criticism)\b',
    r'\b(disputes?|disputed)\b',
    r'\b(argues?\s+against)\b',
    r'\b(contests?|contested)\b',
    r'\b(questions?|questioned)\b',
    r'\b(rejects?|rejected)\b',
    r'\b(however|but|yet|nevertheless)\b',
    r'\b(contrary\s+to|in\s+contrast)\b',
]

SUPPORT_PATTERNS = [
    r'\b(supports?|supported|supporting)\b',
    r'\b(confirms?|confirmed)\b',
    r'\b(demonstrates?|demonstrated)\b',
    r'\b(shows?\s+that|showed\s+that)\b',
    r'\b(establishes?|established)\b',
    r'\b(provides?\s+evidence)\b',
    r'\b(consistent\s+with)\b',
]


def _extract_citation_contexts(text: str, window: int = 300) -> List[Tuple[str, int, str]]:
    results = []
    for m in re.finditer(r'\(([A-Za-z0-9_&.\-]+):\s*p\.(\d+)\)', text):
        doc_id = m.group(1)
        page = int(m.group(2))
        start = max(0, m.start() - window)
        end = min(len(text), m.end() + window)
        context = text[start:end]
        results.append((doc_id, page, context))
    return results


def _infer_context_intent(context: str) -> str:
    ctx = context.lower()
    has_critique = any(re.search(p, ctx) for p in CRITIQUE_PATTERNS)
    has_support = any(re.search(p, ctx) for p in SUPPORT_PATTERNS)
    
    if has_critique and not has_support:
        return "opposing"
    elif has_support and not has_critique:
        return "supporting"
    return "neutral"


def validate_stance_context(text: str, stance_map: Dict[str, str]) -> List[dict]:
    """
    Validate citations are used appropriately given their stance.
    
    Returns list of violations.
    """
    violations = []
    citations = _extract_citation_contexts(text)
    
    for doc_id, page, context in citations:
        doc_stance = stance_map.get(doc_id, "tangential")
        context_intent = _infer_context_intent(context)
        
        if doc_stance == "critiques" and context_intent == "supporting":
            violations.append({
                "doc_id": doc_id,
                "page": page,
                "doc_stance": doc_stance,
                "context_intent": context_intent,
                "mismatch_type": "critiques_as_support",
                "context_snippet": context[:120] + "...",
                "severity": "high"
            })
        elif doc_stance == "supports" and context_intent == "opposing":
            violations.append({
                "doc_id": doc_id,
                "page": page,
                "doc_stance": doc_stance,
                "context_intent": context_intent,
                "mismatch_type": "supports_as_critique",
                "context_snippet": context[:120] + "...",
                "severity": "medium"
            })
    
    return violations


def generate_e4_report(violations: List[dict]) -> str:
    if not violations:
        return "No E4 violations detected."
    
    lines = [f"E4 Violations: {len(violations)}\n"]
    for i, v in enumerate(violations, 1):
        lines.append(f"{i}. [{v['severity'].upper()}] {v['doc_id']} (p.{v['page']})")
        lines.append(f"   Stance: {v['doc_stance']} | Context: {v['context_intent']}")
        lines.append(f"   {v['context_snippet']}")
        lines.append("")
    return "\n".join(lines)


def count_e4_errors(text: str, stance_map: Dict[str, str]) -> Tuple[int, int]:
    """Returns (high_severity_count, total_count)."""
    violations = validate_stance_context(text, stance_map)
    high = len([v for v in violations if v["severity"] == "high"])
    return high, len(violations)
