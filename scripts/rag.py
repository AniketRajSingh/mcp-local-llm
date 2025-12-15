from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from retrieve import retrieve

MODEL_NAME = "google/flan-t5-base"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

def answer(query, k=3):
    chunks = retrieve(query, k)
    context = "\n".join(chunk["content"] for chunk in chunks)

    prompt = f"""
    Answer the question using only the context below.

    Context:
    {context}

    Question:
    {query}
    """

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    outputs = model.generate(**inputs, max_new_tokens=150)

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    print(answer("What is MCP?"))
