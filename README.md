# CampusCompass

Paste a link plus what you're after, straight into the chat box - it goes and
finds the relevant pages on that site, then you can ask questions and get
answers streamed back from your local **Ollama** model. Nothing leaves your
machine.


Example first message:

```
https://www.swinburne.edu.au/ I want course information and research information
```

## How it works

1. **Crawl** (`crawler.py`) - grabs the page you linked, then scores every
   link on it against the words you typed (e.g. "course", "research") and
   only goes after the ones that look relevant. It's a best-link-first crawl,
   not a full site crawl, and it's capped at a small number of pages so it
   stays fast. It also checks `robots.txt` before fetching anything.
2. **Index** (`rag.py`) - chops the crawled text into overlapping chunks and
   embeds each one with Ollama's embeddings API (`nomic-embed-text` by
   default), kept in memory for that session.
3. **Ask** - your question gets embedded too, matched against the chunks by
   cosine similarity, and the best few get handed to your chat model
   (`llama3.1:8b` by default) as context. The reply streams back token by
   token instead of making you wait for the whole thing, with a "Thinking..."
   spinner while it gets started, and shows which page(s) it used underneath.
4. **Chat UI** (`app.py`) - one Streamlit chat box does everything: the first
   link-containing message starts a crawl, everything after that is a
   question against whatever's currently indexed.

<img width="1918" height="904" alt="demo2" src="https://github.com/user-attachments/assets/15a3c455-8a6a-4740-9d0b-aa7611056d21" /> 

<img width="1907" height="848" alt="Screenshot 2026-09-07 204955" src="https://github.com/user-attachments/assets/59f99719-09cb-42cd-a8ca-1f7199cf74e3" />



## Setup

1. Make sure [Ollama](https://ollama.com) is running. This defaults to
   `llama3.1:8b` for chat - if you've already got that pulled, you're set
   there. You'll still need a small embedding model for search, since chat
   models can't do that job:

   ```bash
   ollama pull nomic-embed-text
   ```

2. Install the Python dependencies (a virtual environment is recommended):

   ```bash
   python -m venv venv
   venv\Scripts\activate        # on Windows
   # source venv/bin/activate   # on macOS/Linux

   pip install -r requirements.txt
   ```

3. Run it:

   ```bash
   streamlit run app.py
   ```

   Opens a browser tab at `http://localhost:8501`.

## Using it

1. Type a link plus a topic into the chat box, e.g.
   `https://www.youruni.edu/ I want scholarship and admission information`.
2. Watch the "Looking through the site..." status - it lists each page as it
   reads it, then indexes them.
3. Once it says "Ask away," just type questions normally. Answers stream in
   live and show their sources in a "Sources used" expander underneath.
4. Want a different site or topic? Just paste a new link - it replaces the
   current index and starts fresh.
5. Model settings (Ollama host, chat model, embedding model) live in the
   sidebar under "Settings" if you ever need to change them, along with a
   "Check Ollama connection" button.

## Notes and things you can tweak

- **How the crawl picks pages**: it pulls out meaningful words from what you
  typed (skipping filler like "I want" and "information"), then matches
  those against each link's text and URL. If you don't give it a topic at
  all, it just grabs the first few pages it finds.
- **Speed cap**: `MAX_PAGES` and `MAX_DEPTH` at the top of `crawler.py`
  control how much it's willing to crawl - they default to small numbers (8
  pages, 2 hops) so a crawl finishes quickly. Bump them up if you need more
  coverage and don't mind the wait.
- **Different models**: any Ollama chat model works for "Chat model" (e.g.
  `llama3.1:8b`, `mistral`) as long as you've pulled it. Same for the
  embedding model.
- **Scope**: only follows links on the same domain as the page you gave it,
  and skips non-HTML files (PDFs, images, etc).
- **Politeness**: small delay between requests, and it respects
  `robots.txt`.

## Project files

- `app.py` - the Streamlit chat UI: parses your messages, kicks off crawls,
  and streams answers.
- `crawler.py` - the topic-focused crawler and text extraction.
- `rag.py` - chunking, embedding, and retrieval (cosine similarity).
- `ollama_client.py` - wrapper around Ollama's local REST API, including the
  streaming chat call.
