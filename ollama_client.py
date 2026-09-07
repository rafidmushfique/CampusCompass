from __future__ import annotations

import json

import requests

DEFAULT_HOST = "http://localhost:11434"


class OllamaError(RuntimeError):
    pass


def _base_url(host: str = DEFAULT_HOST) -> str:
    return host.rstrip("/")


def _parse_json(resp, what):
    # ollama should always hand back json here, but if something else is
    # sitting on that port, or it's still starting up, or a proxy/AV is
    # messing with local traffic, we can get back an empty or garbled body
    # instead - this turns that into a message that actually says what
    # happened instead of a raw traceback
    if not resp.text.strip():
        raise OllamaError(
            f"{what}: got an empty reply from {resp.url} (status {resp.status_code}). "
            "Is Ollama actually running, and is anything else using that port?"
        )
    try:
        return resp.json()
    except ValueError as exc:
        snippet = resp.text[:200]
        raise OllamaError(
            f"{what}: that wasn't JSON (status {resp.status_code}): {snippet!r}"
        ) from exc


def list_models(host: str = DEFAULT_HOST) -> list[str]:
    # just pings ollama and returns whatever's pulled locally
    try:
        resp = requests.get(f"{_base_url(host)}/api/tags", timeout=5)
        resp.raise_for_status()
    except requests.exceptions.RequestException as exc:
        raise OllamaError(f"Can't reach Ollama at {host}. Is it running? ({exc})") from exc
    data = _parse_json(resp, "Listing models")
    return [m["name"] for m in data.get("models", [])]


def embed(text: str, model: str = "nomic-embed-text", host: str = DEFAULT_HOST) -> list[float]:
    try:
        resp = requests.post(
            f"{_base_url(host)}/api/embeddings",
            json={"model": model, "prompt": text},
            timeout=60,
        )
        resp.raise_for_status()
    except requests.exceptions.RequestException as exc:
        raise OllamaError(f"Embedding call to '{model}' failed: {exc}") from exc
    data = _parse_json(resp, "Embedding")
    if "embedding" not in data:
        raise OllamaError(f"Ollama didn't return an embedding: {data}")
    return data["embedding"]


def chat_stream(
    messages: list[dict],
    model: str = "llama3.1:8b",
    host: str = DEFAULT_HOST,
    temperature: float = 0.2,
):
    # streams the reply back piece by piece instead of making us sit
    # around for the whole thing - ollama sends one json object per line
    try:
        resp = requests.post(
            f"{_base_url(host)}/api/chat",
            json={
                "model": model,
                "messages": messages,
                "stream": True,
                "options": {"temperature": temperature},
            },
            timeout=180,
            stream=True,
        )
        resp.raise_for_status()
    except requests.exceptions.RequestException as exc:
        raise OllamaError(f"Chat call to '{model}' failed: {exc}") from exc

    got_anything = False
    for line in resp.iter_lines():
        if not line:
            continue
        try:
            chunk = json.loads(line)
        except ValueError as exc:
            raise OllamaError(f"Chat call to '{model}' sent back something odd: {line[:200]!r}") from exc
        got_anything = True
        if chunk.get("done"):
            break
        piece = chunk.get("message", {}).get("content", "")
        if piece:
            yield piece

    if not got_anything:
        raise OllamaError(
            f"Chat call to '{model}' came back empty - check the model name is right "
            f"and pulled (`ollama pull {model}`)."
        )
