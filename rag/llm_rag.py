import os
from typing import Dict, Any, List

from openai import OpenAI
from retrieval.retrieve import retrieve

# You can swap this later (gpt-4o-mini is fast/cheap; gpt-5.2 is stronger)
DEFAULT_MODEL = "gpt-4o-mini"

def _format_context(results: List[Dict[str, Any]]) -> str:
    """
    Build a context block for the LLM with explicit source IDs for citation.
    """
    blocks = []
    for i, r in enumerate(results, start=1):
        m = r["meta"]
        sc_id = m.get("sc_id", "")
        title = m.get("sc_title", "")
        level = m.get("level", "")
        url = m.get("url", "")
        text = r["text"]

        blocks.append(
            f"[S{i}] {sc_id} — {title} (Level {level})\nURL: {url}\n\n{text}\n"
        )
    return "\n---\n".join(blocks)

def _should_refuse(results: List[Dict[str, Any]]) -> bool:
    if not results:
        return True
    # Same heuristic as before; tune as you test
    return results[0]["distance"] > 0.55

def answer_with_llm(question: str, k: int = 5, model: str = DEFAULT_MODEL) -> Dict[str, Any]:
    results = retrieve(question, k=k)

    if _should_refuse(results):
        return {
            "answer": (
                "I can’t answer that confidently from the WCAG 2.2 normative text I retrieved. "
                "Try rephrasing or include the suspected SC number (e.g., 2.4.11)."
            ),
            "citations": [],
            "refused": True,
            "results": results,
        }

    context = _format_context(results)

    system_instructions = (
        "You are a WCAG 2.2 compliance assistant.\n"
        "You MUST answer using only the provided Sources.\n"
        "If the Sources do not contain the answer, say you don’t have enough information.\n"
        "When you make a claim, cite it using the source labels like [S1], [S2].\n"
        "Do not cite anything outside the provided Sources.\n"
    )

    user_prompt = (
        f"Question: {question}\n\n"
        f"Sources:\n{context}\n\n"
        "Write a concise answer. Then include a 'Citations' section listing the cited sources "
        "with SC id + title + URL."
    )

    client = OpenAI()  # uses OPENAI_API_KEY from environment :contentReference[oaicite:2]{index=2}

    # Responses API (recommended for new projects) :contentReference[oaicite:3]{index=3}
    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system_instructions},
            {"role": "user", "content": user_prompt},
        ],
    )

    answer_text = resp.output_text

    # Build a clean citation list from retrieved metadata (top 3 by default)
    citations = []
    for r in results[:3]:
        m = r["meta"]
        citations.append({
            "sc_id": m.get("sc_id", ""),
            "sc_title": m.get("sc_title", ""),
            "level": m.get("level", ""),
            "url": m.get("url", ""),
        })

    return {
        "answer": answer_text,
        "citations": citations,
        "refused": False,
        "results": results,
        "model": model,
    }
