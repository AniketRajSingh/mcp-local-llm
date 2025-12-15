import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

ARTIFACT_DIR = "artifacts"
EMBED_MODEL = "all-MiniLM-L6-v2"

embedder = SentenceTransformer(EMBED_MODEL)

def load_artifacts():
    index = faiss.read_index(f"{ARTIFACT_DIR}/faiss.index")
    with open(f"{ARTIFACT_DIR}/metadata.json") as f:
        metadata = json.load(f)
    return index, metadata

def retrieve(query, k=3):
    index, metadata = load_artifacts()
    q_emb = embedder.encode([query]).astype("float32")
    _, idx = index.search(q_emb, k)
    return [metadata[i] for i in idx[0]]

if __name__ == "__main__":
    results = retrieve("What is MCP?")
    print(results)
