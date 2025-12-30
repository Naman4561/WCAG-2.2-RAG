import json
import os
from typing import List, Dict

import chromadb
from sentence_transformers import SentenceTransformer

INPUT_PATH = os.path.join("data", "processed", "wcag22_spec_sc.jsonl")
PERSIST_DIR = os.path.join("data", "vectorstore", "chroma_wcag22")
COLLECTION_NAME = "wcag22_spec"

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def load_jsonl(path: str) -> List[Dict]:
    docs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    return docs

def main():
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(
            f"Missing {INPUT_PATH}. Run ingestion/parse_spec.py first."
        )

    os.makedirs(PERSIST_DIR, exist_ok=True)

    docs = load_jsonl(INPUT_PATH)
    print(f"[INFO] Loaded {len(docs)} chunks")

    model = SentenceTransformer(EMBED_MODEL_NAME)

    client = chromadb.PersistentClient(path=PERSIST_DIR)

    # Reset collection during development
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    texts = [d["text"] for d in docs]
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    ids = [d["sc_id"] for d in docs]
    def safe_str(x):
        return "" if x is None else str(x)

    metadatas = [
        {
        "sc_id": safe_str(d.get("sc_id")),
        "sc_title": safe_str(d.get("sc_title")),
        "level": safe_str(d.get("level")),  
        "url": safe_str(d.get("url")),
        "source": safe_str(d.get("source")),
        "normativity": safe_str(d.get("normativity")),
        "version": safe_str(d.get("version")),
        }
        for d in docs
    ]   
    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings.tolist(),
        metadatas=metadatas
    )

    print(
        f"[INFO] Index built at {PERSIST_DIR} "
        f"(collection='{COLLECTION_NAME}', items={collection.count()})"
    )

if __name__ == "__main__":
    main()
