import json
import os
import re
from typing import Optional, Tuple, List
from bs4 import BeautifulSoup, Tag

print("=== parse_spec.py is running ===")

RAW_SPEC_PATH = os.path.join("data", "raw", "www.w3.org", "TR", "WCAG22", "index.html")
OUT_PATH = os.path.join("data", "processed", "wcag22_spec_sc.jsonl")

SC_LABEL_RE = re.compile(
    r"^\s*Success Criterion\s+(\d+\.\d+\.\d+)\s+(.+?)\s*$",
    re.IGNORECASE
)
SC_ANYWHERE_RE = re.compile(r"\b(\d+\.\d+\.\d+)\b")

LEVEL_RE = re.compile(r"\bLevel\s+(A{1,3})\b", re.IGNORECASE)

def normalize_text(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def parse_sc_from_text(txt: str) -> Optional[Tuple[str, str]]:
    m = SC_LABEL_RE.match(txt)
    if m:
        return m.group(1), m.group(2)

    m2 = SC_ANYWHERE_RE.search(txt)
    if m2:
        sc_id = m2.group(1)
        # Drop conformance subsection matches like 5.x.x
        if sc_id.startswith("5."):
            return None
        title = txt.split(sc_id, 1)[-1].strip(" :-–—")
        if title:
            return sc_id, title
    return None

def is_sc_node(tag: Tag) -> Optional[Tuple[str, str]]:
    # SCs might appear in headings or <dt> terms
    if tag.name in ("h2", "h3", "h4", "h5", "dt"):
        txt = tag.get_text(" ", strip=True)
        return parse_sc_from_text(txt)
    return None

def is_major_section_heading(tag: Tag) -> bool:
    """True if this tag is a top-level section heading (used to stop the last SC from eating later sections)."""
    return tag.name == "h2"

def collect_until_next_sc(start_node: Tag) -> str:
    parts: List[str] = []

    # include the title line
    parts.append(start_node.get_text(" ", strip=True))

    for el in start_node.next_elements:
        if isinstance(el, Tag):
            if el is not start_node and is_sc_node(el):
                break
            if el is not start_node and is_major_section_heading(el):
                break

            # skip nav-like sections
            if el.name in ("nav", "header", "footer"):
                continue

            # collect readable text blocks
            if el.name in ("p", "li", "dt", "dd", "blockquote"):
                txt = el.get_text(" ", strip=True)
                if txt:
                    parts.append(txt)

    return normalize_text("\n".join(parts))

def infer_level(text: str) -> Optional[str]:
    m = LEVEL_RE.search(text)
    return m.group(1).upper() if m else None

def anchor_for(tag: Tag) -> Optional[str]:
    # best effort: use tag id or parent id
    if tag.get("id"):
        return tag["id"]
    if tag.parent and tag.parent.get("id"):
        return tag.parent["id"]
    return None

def main() -> None:
    print(f"[INFO] CWD: {os.getcwd()}")
    print(f"[INFO] Reading: {RAW_SPEC_PATH}")
    print(f"[INFO] Writing: {OUT_PATH}")

    if not os.path.exists(RAW_SPEC_PATH):
        print("[ERROR] RAW_SPEC_PATH does not exist.")
        return

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    with open(RAW_SPEC_PATH, "rb") as f:
        soup = BeautifulSoup(f.read(), "html.parser")

    # Collect SC nodes in document order
    sc_nodes: List[Tuple[Tag, str, str]] = []
    for tag in soup.find_all(["h2", "h3", "h4", "h5", "dt"]):
        parsed = is_sc_node(tag)
        if parsed:
            sc_nodes.append((tag, parsed[0], parsed[1]))

    # Deduplicate by sc_id while preserving order
    seen = set()
    ordered = []
    for tag, sc_id, sc_title in sc_nodes:
        if sc_id in seen:
            continue
        seen.add(sc_id)
        ordered.append((tag, sc_id, sc_title))

    print(f"[INFO] SC nodes found (deduped): {len(ordered)}")
    print("[INFO] First 10 SCs:")
    for _, sc_id, sc_title in ordered[:10]:
        print(f"  - {sc_id} {sc_title}")

    chunks = []
    for tag, sc_id, sc_title in ordered:
        text = collect_until_next_sc(tag)
        level = infer_level(text)
        anchor_id = anchor_for(tag)
        url = f"https://www.w3.org/TR/WCAG22/#{anchor_id}" if anchor_id else "https://www.w3.org/TR/WCAG22/"

        chunks.append({
            "doc_set": "wcag22",
            "source": "wcag_spec",
            "normativity": "normative",
            "version": "2.2",
            "sc_id": sc_id,
            "sc_title": sc_title,
            "level": level,
            "url": url,
            "text": text
        })

    with open(OUT_PATH, "w", encoding="utf-8") as out:
        for c in chunks:
            out.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"[INFO] Wrote {len(chunks)} chunks -> {OUT_PATH}")

if __name__ == "__main__":
    main()
