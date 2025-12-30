from rag.spec_rag import answer

if __name__ == "__main__":
    while True:
        q = input("\nAsk a WCAG question (or 'quit'): ").strip()
        if q.lower() in ("quit", "exit"):
            break

        out = answer(q, k=5)
        print("\n--- ANSWER ---")
        print(out["answer"])

        print("\n--- CITATIONS ---")
        for c in out["citations"]:
            print(f"- {c['sc_id']} ({c['level']}) — {c['sc_title']} | {c['url']}")

        print("\nRefused:", out["refused"])
