from typing import Dict, Any, List
from retrieval.retrieve import retrieve

def citation(meta: Dict[str, Any]) -> Dict[str, str]:
    return {
        "sc_id": meta.get("sc_id", ""),
        "sc_title": meta.get("sc_title", ""),
        "level": meta.get("level", ""),
        "url": meta.get("url", ""),
    }

def should_refuse(results: List[Dict[str, Any]]) -> bool:
    # cosine distance: lower = more similar
    if not results:
        return True
    best = results[0]["distance"]
    # Start with this; tune after you see behavior
    return best > 0.40

def build_answer_from_top_result(question: str, top: Dict[str, Any]) -> str:
    meta = top["meta"]
    sc_id = meta.get("sc_id", "Unknown")
    title = meta.get("sc_title", "")
    level = meta.get("level", "Unknown")

    # Keep it grounded: we *quote/excerpt* rather than freestyle.
    text = top["text"]
    excerpt = text[:1200] + ("…" if len(text) > 1200 else "")

    return (
        f"Based on the WCAG 2.2 normative text I retrieved, the most relevant requirement is:\n"
        f"**{sc_id} — {title} (Level {level})**\n\n"
        f"**Normative excerpt:**\n{excerpt}"
    )

def answer(question: str, k: int = 5) -> Dict[str, Any]:
    results = retrieve(question, k=k)

    if should_refuse(results):
        return {
            "answer": (
                "I can’t answer that confidently from the WCAG 2.2 normative text I retrieved. "
                "Try rephrasing or mention the relevant success criterion number (e.g., 2.4.11)."
            ),
            "citations": [],
            "refused": True,
            "results": results,
        }

    top = results[0]
    return {
        "answer": build_answer_from_top_result(question, top),
        "citations": [citation(r["meta"]) for r in results[:3]],
        "refused": False,
        "results": results,
    }
