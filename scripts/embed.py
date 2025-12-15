import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from ingest import ingest

ARTIFACT_DIR = "artifacts"
EMBED_MODEL = "all-MiniLM-L6-v2"

def build_index():
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    chunks, sources = ingest()

    embedder = SentenceTransformer(EMBED_MODEL)
    embeddings = embedder.encode(chunks, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    metadata = []
    for idx, (chunk, src) in enumerate(zip(chunks, sources)):
        metadata.append({
            "id": idx,
            "content": chunk,
            "source": src
        })

    faiss.write_index(index, f"{ARTIFACT_DIR}/faiss.index")

    with open(f"{ARTIFACT_DIR}/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("Artifacts created successfully")

if __name__ == "__main__":
    build_index()
