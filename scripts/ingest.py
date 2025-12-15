import os
from transformers import AutoTokenizer

DATA_DIR = "data/raw"
MODEL_NAME = "bert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def load_documents():
    documents = []
    sources = []

    for file in os.listdir(DATA_DIR):
        path = os.path.join(DATA_DIR, file)
        if os.path.isfile(path) and file.endswith(".txt"):
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                documents.append(f.read())
                sources.append(file)

    return documents, sources

def chunk_text(text, max_tokens=400, overlap=50):
    tokens = tokenizer.encode(text)
    step = max_tokens - overlap

    for i in range(0, len(tokens), step):
        yield tokenizer.decode(tokens[i:i + max_tokens])

def ingest():
    docs, sources = load_documents()

    chunks = []
    chunk_sources = []

    for doc, src in zip(docs, sources):
        for chunk in chunk_text(doc):
            chunks.append(chunk)
            chunk_sources.append(src)

    return chunks, chunk_sources

if __name__ == "__main__":
    chunks, sources = ingest()
    print(f"Chunks created: {len(chunks)}")
