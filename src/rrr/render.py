import re


CITE_RE = re.compile(r"\(([A-Za-z0-9_&.\-]+):\s*p\.(\d+)\)")


def render_citation(doc_id, page) -> str:
    return f"({str(doc_id).strip()}: p.{int(page)})"


def parse_citations(text: str):
    for match in CITE_RE.finditer(text or ""):
        yield {
            "doc_id": match.group(1),
            "page": int(match.group(2)),
            "start": match.start(),
            "end": match.end(),
            "raw": match.group(0),
        }


def render_markdown(obj, refs_by_id):
    lines = []
    lines.append(f"**Claim/Topic**: {obj.get('claim') or obj.get('topic','')}")
    lines.append("")
    lines.append("**Evidence (snippets)**:")
    for e in obj.get("evidence", []):
        tag = "Quote" if e.get("type")=="quote" else "Paraphrase"
        doc_id = e.get("doc_id")
        ref = refs_by_id.get(doc_id, doc_id)
        text = (e.get("text", "") or "")
        snippet = text[:180].replace("\n", " ")
        page = e.get("page")
        lines.append(f"- {tag}: {snippet} {render_citation(doc_id, page)}")
        if ref and ref != doc_id:
            lines.append(f"  Source: {ref}")
    return "\n".join(lines)
