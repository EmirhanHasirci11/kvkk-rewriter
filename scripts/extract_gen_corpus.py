"""Text of the generation-only PDFs: gen_corpus/raw/*.pdf -> gen_corpus/text/*.txt. No LLM call.

Run from the repo root: python scripts/extract_gen_corpus.py

A generic version of kvkk-rag's scripts/extract_corpus.py (that one is tuned per document and runs on
import): PyMuPDF blocks, spans outside the body font size range dropped (footnotes, page numbers, running
headers), table-of-contents pages and entries, covers and publisher notes dropped, words hyphenated across
lines joined, ligatures replaced, NFC.
The texts only seed question generation, so the bar is "readable passages", not retrieval-grade cleaning.
Writes gen_corpus/manifest.json (pages, words per document). A PDF without a text layer (scanned) is listed
there as skipped; there is no OCR.
"""
import collections
import json
import re
import sys
import unicodedata

import pymupdf

import kvkk_rag

RAW = kvkk_rag.ROOT / "gen_corpus/raw"
TEXT = kvkk_rag.ROOT / "gen_corpus/text"
PAGENUM_RE = re.compile(r"^\s*([0-9]{1,3}|[ivxlcdm]{1,6})\s*$", re.I)
TOC_RE = re.compile(r"İÇİNDEKİLER|İçindekiler")
# a table-of-contents entry inside a page: a title followed by a page number and the next numbered title
TOC_ENTRY_RE = re.compile(r"\s\d{1,3}\s+(?:[A-ZÇĞİÖŞÜ]\.|[IVX]+\.|\d+(?:\.\d+)*\.?\s)")
BOILERPLATE_RE = re.compile(r"tüm hakları saklıdır|izin alınmadan kısmen|Bu kitapta yer alan|Fikir ve Sanat Eserleri"
                            r"|Faks:|Telefon:|Tel:|KVKK YAYINLARI|Yayınları No", re.I)
# ligatures, soft hyphen, zero-width space
LIGATURES = {chr(0xFB01): "fi", chr(0xFB02): "fl", chr(0xFB00): "ff", chr(0xFB03): "ffi", chr(0xFB04): "ffl", chr(0xAD): "", chr(0x200B): ""}


def is_noise(para: str) -> bool:
    """Covers, publisher notes and table-of-contents runs that survived the page-level filter."""
    return bool(BOILERPLATE_RE.search(para)) or len(TOC_ENTRY_RE.findall(para)) >= 3


def body_size(doc) -> float:
    c = collections.Counter()
    for page in doc:
        for b in page.get_text("dict")["blocks"]:
            for line in b.get("lines", []):
                for s in line["spans"]:
                    c[round(s["size"], 1)] += len(s["text"].strip())
    return c.most_common(1)[0][0] if c else 0.0


def join_lines(lines: list[str]) -> str:
    out = ""
    for ln in (x.strip() for x in lines):
        if not ln:
            continue
        if out and re.search(r"[a-zçğıöşü]-$", out) and re.match(r"^[a-zçğıöşü]", ln):
            out = out[:-1] + ln
        else:
            out = f"{out} {ln}" if out else ln
    return re.sub(r"[ \t ]+", " ", out).strip()


def extract(path) -> tuple[str, int]:
    doc = pymupdf.open(path)
    bs = body_size(doc)
    if not bs:
        return "", doc.page_count     # scanned images, no text layer (no OCR here)
    paras = []
    for page in doc:
        txt = page.get_text()
        if TOC_RE.search(txt) or txt.count("....") > 5:
            continue
        for b in page.get_text("dict")["blocks"]:
            lines = ["".join(s["text"] for s in line["spans"] if bs * 0.9 <= s["size"] <= bs * 3)
                     for line in b.get("lines", [])]
            para = join_lines(lines)
            if para and not PAGENUM_RE.match(para) and len(para.split()) > 2 and not is_noise(para):
                paras.append(para)
    # merge a paragraph into the previous one when the previous does not end a sentence
    merged = []
    for p in paras:
        if merged and (re.match(r"^[a-zçğıöşü]", p) or not re.search(r"[.:;!?”\"]$", merged[-1])):
            merged[-1] = join_lines([merged[-1], p])
        else:
            merged.append(p)
    # checked again after merging: a contents run or a cover note is often split over several blocks
    merged = [p for p in merged if not is_noise(p)]
    text = unicodedata.normalize("NFC", "\n\n".join(merged))
    for a, b in LIGATURES.items():
        text = text.replace(a, b)
    return text.strip(), doc.page_count


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    sources = json.loads((kvkk_rag.ROOT / "gen_corpus/sources.json").read_text(encoding="utf-8"))
    TEXT.mkdir(parents=True, exist_ok=True)
    manifest = []
    for d in sources["documents"]:
        text, pages = extract(RAW / f"{d['id']}.pdf")
        (TEXT / f"{d['id']}.txt").write_text(text + "\n", encoding="utf-8")
        manifest.append({"id": d["id"], "title": d["title"], "url": d["url"], "sha256": d["sha256"],
                         "pages": pages, "words": len(text.split())})
        print(f"{d['id']} {pages:4d} pages {len(text.split()):7d} words  {d['title'][:70]}")
    (kvkk_rag.ROOT / "gen_corpus/manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1),
                                                         encoding="utf-8")
    print(f"total {sum(m['words'] for m in manifest)} words in {len(manifest)} documents")


if __name__ == "__main__":
    main()
