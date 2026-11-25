from pathlib import Path


FILLER = (
    "This synthetic working paper exists only to exercise the ingest cascade. "
    "It discusses labour recruitment, seasonal migration, taxation, transport "
    "costs and household allocation in a stylised colonial economy, and it "
    "repeats its argument at length so that the opening pages carry enough "
    "text to pass the usable-text check."
)


def make_pdf(path: Path, title: str, authors: str, year: str, extra: str = "") -> Path:
    import pymupdf

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open()
    page = doc.new_page()
    body = f"{title}\n\n{authors}\n\n{year}\n\n{FILLER}\n\n{extra}".strip()
    page.insert_textbox(
        pymupdf.Rect(60, 60, 540, 780), body, fontsize=10, fontname="helv",
    )
    doc.save(str(path))
    doc.close()
    return path


def write_bib(path: Path, entries: list) -> Path:
    blocks = []
    for entry_type, key, fields in entries:
        body = ",\n".join(f"  {name} = {{{value}}}" for name, value in fields.items())
        blocks.append(f"@{entry_type}{{{key},\n{body}\n}}")
    path = Path(path)
    path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
    return path
