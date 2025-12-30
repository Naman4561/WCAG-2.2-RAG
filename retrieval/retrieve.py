import os
from typing import List, Dict, Any
import chromadb
from sentence_transformers import SentenceTransformer

PERSIST_DIR = os.path.join("data", "vectorstore", "chroma_wcag22")
COLLECTION_NAME = "wcag22_spec"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

_model = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL_NAME)
    return _model

def retrieve(query: str, k: int = 5) -> List[Dict[str, Any]]:
    client = chromadb.PersistentClient(path=PERSIST_DIR)
    col = client.get_collection(COLLECTION_NAME)

    model = get_model()
    q_emb = model.encode([query], normalize_embeddings=True)[0].tolist()

    res = col.query(
        query_embeddings=[q_emb],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    results = []
    for i in range(len(res["ids"][0])):
        results.append({
            "id": res["ids"][0][i],
            "distance": res["distances"][0][i],
            "text": res["documents"][0][i],
            "meta": res["metadatas"][0][i],
        })
    return results
