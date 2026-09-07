import re

import numpy as np
import streamlit as st

from crawler import crawl_topic
from ollama_client import OllamaError, chat_stream, embed, list_models
from rag import build_prompt, chunk_pages, embed_chunks, top_k_chunks

st.set_page_config(page_title="CampusCompass", page_icon="🧭", layout="wide")

st.title("🧭 CampusCompass")
st.caption(
    "Drop a link and say what you're after, e.g. \"https://www.swinburne.edu.au/ "
    "I want course information and research information\" - it'll go find the "
    "relevant pages, then you can just ask questions."
)

URL_RE = re.compile(r"https?://\S+")

# first run of the session - set up the bits we track across reruns
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.chunks = []
    st.session_state.chunk_vectors = None
    st.session_state.pages = []
    st.session_state.start_url = None
    st.session_state.topic = None

with st.sidebar:
    with st.expander("Settings", expanded=False):
        ollama_host = st.text_input("Ollama host", value="http://localhost:11434")
        llm_model = st.text_input("Chat model", value="llama3.1:8b")
        embed_model = st.text_input("Embedding model", value="nomic-embed-text")
        if st.button("Check Ollama connection"):
            try:
                models = list_models(ollama_host)
                st.success(f"Connected: {', '.join(models) if models else '(no models pulled yet)'}")
            except OllamaError as e:
                st.error(str(e))

    if st.session_state.start_url:
        st.subheader("Currently indexed")
        st.write(st.session_state.start_url)
        if st.session_state.topic:
            st.caption(f"Topic: {st.session_state.topic}")
        with st.expander(f"{len(st.session_state.pages)} page(s)"):
            for p in st.session_state.pages:
                st.write(f"- [{p.title}]({p.url})")

# replay whatever's already been said before handling anything new
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


def do_crawl(url, topic):
    with st.chat_message("assistant"):
        status = st.status("Looking through the site...", expanded=True)

        def report(line):
            status.write(line)

        try:
            pages = crawl_topic(url, topic, status_callback=report)
        except Exception as e:
            status.update(label="Something went wrong", state="error")
            st.error(f"Couldn't crawl that: {e}")
            return

        if not pages:
            status.update(label="No pages found", state="error")
            st.warning("Couldn't find anything readable there - double check the link.")
            return

        status.update(label=f"Making sense of {len(pages)} page(s)...")
        chunks = chunk_pages(pages)

        def report_embed(done, total):
            status.write(f"Indexing chunk {done}/{total}")

        try:
            vectors = embed_chunks(chunks, model=embed_model, host=ollama_host, progress_callback=report_embed)
        except OllamaError as e:
            status.update(label="Indexing failed", state="error")
            st.error(f"{e}\n\nMake sure you've pulled the embedding model: `ollama pull {embed_model}`")
            return

        # swap in the new site - old index is gone once this lands
        st.session_state.pages = pages
        st.session_state.chunks = chunks
        st.session_state.chunk_vectors = vectors
        st.session_state.start_url = url
        st.session_state.topic = topic

        status.update(label=f"Indexed {len(pages)} page(s)", state="complete", expanded=False)

    # they already told us what they're after in the same message, so
    # answer that straight away instead of making them ask it again
    if topic:
        do_answer(topic)
    else:
        with st.chat_message("assistant"):
            msg = f"Went through {len(pages)} page(s) on {url}. Ask away."
            st.markdown(msg)
        st.session_state.messages.append({"role": "assistant", "content": msg})


def do_answer(question):
    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking..."):
                q_vec = np.array(embed(question, model=embed_model, host=ollama_host), dtype=np.float32)
                relevant = top_k_chunks(q_vec, st.session_state.chunk_vectors, st.session_state.chunks, k=5)
                prompt = build_prompt(question, relevant)
                stream = chat_stream(prompt, model=llm_model, host=ollama_host)
                # pulling the first piece here is what keeps the spinner up
                # until the model actually starts talking
                first_piece = next(stream, "")

            def full_stream():
                if first_piece:
                    yield first_piece
                yield from stream

            # the model explains which page(s) it used right in the
            # answer text now, so there's no separate link list to build
            answer = st.write_stream(full_stream())

            st.session_state.messages.append({"role": "assistant", "content": answer})
        except OllamaError as e:
            st.error(str(e))
            st.session_state.messages.append({"role": "assistant", "content": f"Error: {e}"})


placeholder = "e.g. https://www.swinburne.edu.au/ I want course information and research information"
user_input = st.chat_input(placeholder)

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    found = URL_RE.findall(user_input)
    if found:
        # a link showed up - treat this as "go crawl that site"
        url = found[0]
        topic = user_input.replace(url, "").strip()
        do_crawl(url, topic)
    elif not st.session_state.chunks:
        # no site indexed yet and no link given - nothing to answer from
        with st.chat_message("assistant"):
            nudge = (
                "Give me a link to start with, like: `https://www.swinburne.edu.au/ "
                "I want course information and research information`"
            )
            st.markdown(nudge)
        st.session_state.messages.append({"role": "assistant", "content": nudge})
    else:
        do_answer(user_input)
