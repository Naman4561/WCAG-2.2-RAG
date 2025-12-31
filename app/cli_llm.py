from rag.llm_rag import answer_with_llm

if __name__ == "__main__":
    while True:
        q = input("\nAsk a WCAG question (or 'quit'): ").strip()
        if q.lower() in ("quit", "exit"):
            break

        out = answer_with_llm(q, k=5, model="gpt-4o-mini")
        print("\n--- ANSWER ---")
        print(out["answer"])

        print("\n--- TOP METADATA (debug) ---")
        for c in out["citations"]:
            print(f"- {c['sc_id']} ({c['level']}) — {c['sc_title']} | {c['url']}")
