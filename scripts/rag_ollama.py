"""Ollama-backed RAG answer function.

This module loads retrieval artifacts via `retrieve.retrieve`, builds a prompt
from retrieved chunks, and sends the prompt to a local Ollama server
(`http://localhost:11434`) to generate a response.

The implementation is intentionally small and defensive:
- If Ollama is unreachable, returns an informative message.
- If Ollama returns JSON, tries to extract text from common fields.
- Falls back to returning raw response text.

You can later replace or extend the `generate_with_ollama` function to
support streaming, authentication, or advanced prompt formatting.
"""

import json
import requests
from retrieve import retrieve

OLLAMA_URL = "http://localhost:11434"
OLLAMA_GENERATE_ENDPOINT = f"{OLLAMA_URL}/api/generate"
OLLAMA_TAGS_ENDPOINT = f"{OLLAMA_URL}/api/tags"


def _choose_model():
    """Try to discover a running Ollama model. Returns a model name or None."""
    try:
        resp = requests.get(OLLAMA_TAGS_ENDPOINT, timeout=2)
        if resp.status_code == 200:
            data = resp.json()
            # data may include a 'models' list of objects with 'name'
            models = []
            if isinstance(data, dict):
                # try common keys
                if "models" in data and isinstance(data["models"], list):
                    models = [m.get("name") for m in data["models"] if isinstance(m, dict) and m.get("name")]
                elif "tags" in data and isinstance(data["tags"], list):
                    # fallback if tags listing is returned
                    models = data["tags"]
            elif isinstance(data, list):
                # maybe already a list of model names
                models = data

            if models:
                return models[0]
    except Exception:
        pass
    return None


def generate_with_ollama(prompt: str, model: str = None, max_tokens: int = 150) -> str:
    """Send the prompt to the local Ollama server and return generated text.

    Args:
        prompt: The full prompt string to send to Ollama.
        model: Optional Ollama model name. If None, attempts to auto-discover.
        max_tokens: Max tokens to request (best-effort; Ollama may ignore).

    Returns:
        Generated text (string) or an error message.
    """
    if model is None:
        model = _choose_model()

    payload = {"prompt": prompt}
    if model:
        payload["model"] = model
    # include max_tokens where supported
    payload["max_tokens"] = max_tokens

    try:
        resp = requests.post(OLLAMA_GENERATE_ENDPOINT, json=payload, timeout=15)
    except requests.exceptions.RequestException as e:
        return f"[ollama connection error] Could not reach Ollama at {OLLAMA_URL}: {e}"

    # Try to parse JSON responses first
    content = None
    ct = resp.headers.get("Content-Type", "")
    if "application/json" in ct:
        try:
            data = resp.json()
            # Common places for generated text in different Ollama response shapes
            if isinstance(data, dict):
                # Look for 'result' or 'results'
                if "result" in data and isinstance(data["result"], str):
                    content = data["result"]
                elif "results" in data and isinstance(data["results"], list):
                    # join text fields from results
                    parts = []
                    for r in data["results"]:
                        if isinstance(r, dict):
                            # known field names
                            for k in ("content", "text", "message", "generated"):
                                if k in r and isinstance(r[k], str):
                                    parts.append(r[k])
                        elif isinstance(r, str):
                            parts.append(r)
                    if parts:
                        content = "\n".join(parts)
                # some responses embed generation under 'response' or 'output'
                elif "response" in data and isinstance(data["response"], dict):
                    # try to gather text
                    resp_obj = data["response"]
                    for k in ("generated", "text", "content", "output"):
                        if k in resp_obj and isinstance(resp_obj[k], str):
                            content = resp_obj[k]
                            break
            # fallback: stringify the JSON
            if content is None:
                content = json.dumps(data)
        except Exception:
            content = resp.text
    else:
        # Non-JSON: could be plain text or NDJSON streaming concatenated
        text = resp.text
        # If NDJSON (multiple JSON objects on separate lines), try to extract text fields
        lines = [l for l in text.splitlines() if l.strip()]
        extracted = []
        for line in lines:
            try:
                obj = json.loads(line)
                # try same extraction
                if isinstance(obj, dict):
                    for k in ("content", "text", "message", "generated", "output"):
                        if k in obj and isinstance(obj[k], str):
                            extracted.append(obj[k])
            except Exception:
                # not JSON, maybe plain text
                extracted.append(line)
        if extracted:
            content = "\n".join(extracted)
        else:
            content = text

    return content


def answer(query: str, k: int = 3, model: str = None, max_tokens: int = 150):
    """Retrieve top-k chunks and ask Ollama to answer using the retrieved context.

    Args:
        query: User question.
        k: Number of retrieved chunks to include as context.
        model: Optional Ollama model name to use.
        max_tokens: Max tokens for generation.

    Returns:
        str: Generated answer or diagnostic message.
    """
    chunks = retrieve(query, k)
    context = "\n".join(chunk.get("content", "") for chunk in chunks)

    prompt = f"""
Answer the question using only the context below. If the answer is unknown, say you don't know.

Context:
{context}

Question:
{query}
"""

    return generate_with_ollama(prompt, model=model, max_tokens=max_tokens)


if __name__ == "__main__":
    print(answer("What is Retrieval Augmented Generation?"))
