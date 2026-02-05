def render_markdown(obj, refs_by_id):
    lines = []
    lines.append(f"**Claim/Topic**: {obj.get('claim') or obj.get('topic','')}")
    lines.append("")
    lines.append("**Evidence (snippets)**:")
    for e in obj.get("evidence", []):
        tag = "Quote" if e.get("type")=="quote" else "Paraphrase"
        ref = refs_by_id.get(e.get("doc_id"), e.get("doc_id"))
        text = (e.get("text", "") or "")
        snippet = text[:180].replace("\n", " ")
        page = e.get("page")
        lines.append(f"- {tag}: {snippet} [{ref}: p.{page}]")
    return "\n".join(lines)
