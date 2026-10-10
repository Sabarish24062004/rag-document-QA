# RAG Document Q&A

Upload PDFs, ask questions in plain language, and get answers grounded in your documents, with citations that link to the exact page.

## How it works

```
 INDEXING (on upload)
 PDF ─► extract text ─► split into chunks ─► chunks → vectors ─► store in vector DB
        (PyMuPDF,        (≈1000 chars,        (all-MiniLM-L6-v2,   (ChromaDB, cosine,
         per page)        200 overlap,         local, 384-dim)       on disk)
                          page kept)

 QUESTION ANSWERING (on ask)
 question ─► question → vector ─► find most similar chunks ─► LLM writes answer ─► citations
                                   (top-k, optional            from those chunks     [1] → file, page,
                                    document filter)           only                  passage, link
```

| Step | Code |
| --- | --- |
| Extract text page by page | [app/rag/pdf_loader.py](app/rag/pdf_loader.py) |
| Split into chunks | [app/rag/chunker.py](app/rag/chunker.py) |
| Text → vectors | [app/rag/embeddings.py](app/rag/embeddings.py) |
| Store / search vectors | [app/rag/vector_store.py](app/rag/vector_store.py) |
| Generate answer | [app/rag/llm.py](app/rag/llm.py) |
| Orchestration + citations | [app/rag/pipeline.py](app/rag/pipeline.py) |
| REST API | [app/api/routes.py](app/api/routes.py) |
| Web UI | [frontend/](frontend/) |

## Setup

Requires Python 3.10+.

```powershell
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env          # macOS/Linux: cp .env.example .env
```

### Free AI answers with Google Gemini

1. Open https://aistudio.google.com/apikey and sign in with a Google account.
2. Click **Create API key** and copy it.
3. In `.env`, set `GEMINI_API_KEY=<your key>`.
4. Restart the server.

The Gemini free tier needs no credit card. It has per-minute and per-day request limits, and Google may use free-tier content to improve its products, so avoid uploading confidential PDFs. Change `GEMINI_MODEL` to try another model.

Without a key the app still works in **passage mode**: it shows the most relevant passages with citations instead of writing an answer.

Other OpenAI-compatible providers (OpenAI, Groq, Ollama) can be used via `LLM_BASE_URL`, `LLM_API_KEY` and `LLM_MODEL` when `GEMINI_API_KEY` is empty.

## Run

```powershell
uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000. The first upload downloads the embedding model (about 80 MB, once) to `~/.cache/chroma`.

Interactive API docs: http://127.0.0.1:8000/docs

## Host it online (Docker)

The repo includes a `Dockerfile` and a `docker-compose.yml` that runs the app behind [Caddy](https://caddyserver.com), which obtains a free HTTPS certificate automatically. HTTPS is required for installing the app on phones.

You need a Linux server with Docker (for example a DigitalOcean, Hetzner, AWS Lightsail or Oracle Cloud VM with at least 2 GB RAM) and a domain name.

1. Point a DNS **A record** for your domain (e.g. `docs.example.com`) at the server's IP. Open ports 80 and 443 in the server firewall.
2. Copy the project to the server and create `.env` from `.env.example`. Set:
   - `DOMAIN=docs.example.com`
   - `APP_PASSWORD=<a long password>`: **required** once the app is online. Without it anyone with the link can read your PDFs and use your Gemini quota.
   - `GEMINI_API_KEY=<your key>`
3. Start it:
   ```bash
   docker compose up -d --build
   ```
4. Open `https://docs.example.com` and log in.

Uploaded PDFs and the vector index live in Docker volumes, so they survive restarts and rebuilds. To update after code changes: `docker compose up -d --build`. Logs: `docker compose logs -f app`.

Without a domain, run only the app container and use plain HTTP (the app works, but phones won't offer to install it):
```bash
docker build -t doc-qa .
docker run -d -p 8000:8000 --env-file .env -v doc-qa-docs:/app/data/documents -v doc-qa-db:/app/chroma_db doc-qa
```

## Install as a phone or laptop app

The site is a Progressive Web App: once it is served over HTTPS you can install it like a regular app, with its own icon and window.

| Device | How to install |
| --- | --- |
| Android (Chrome) | Open the site → tap **Install app** in the header, or menu ⋮ → **Install app** |
| iPhone / iPad (Safari) | Open the site → Share button → **Add to Home Screen** |
| Windows / Mac / Linux (Chrome, Edge) | Click **Install app** in the header, or the install icon in the address bar |

Updates are picked up automatically the next time the app is opened. The interface opens offline, but asking questions needs a connection to the server.

## Using the app

1. Drag PDFs onto the upload box (or click it). Each file is extracted, chunked and indexed; uploading a file with the same name replaces it.
2. Type a question and press Enter.
3. Click a citation chip such as **[1]** to jump to its source passage, then **Open page** to view that page of the PDF.
4. Tick documents in the sidebar to search only those; untick all to search everything.
5. Change **Sources to retrieve** to give the model more or less context.

## API

| Method | Path | Description |
| --- | --- | --- |
| GET | `/api/status` | LLM mode, model, document count |
| POST | `/api/documents` | Upload a PDF (multipart `file`) |
| GET | `/api/documents` | List indexed documents |
| DELETE | `/api/documents/{filename}` | Remove a document and its vectors |
| GET | `/api/documents/{filename}/file` | Download/view the PDF |
| POST | `/api/ask` | `{"question": "...", "top_k": 5, "documents": ["a.pdf"]}` |
| GET | `/api/session` | Whether a password is required and the caller is logged in |
| POST | `/api/login` / `/api/logout` | `{"password": "..."}` sets / clears the session cookie |
| GET | `/health` | Health check |

When `APP_PASSWORD` is set, all other `/api` routes and `/docs` return 401 until you log in.

`/api/ask` returns the answer plus `citations`: number, filename, page, passage text, similarity score, and whether the answer cited it.

## Tests

```powershell
python -m pytest
```

Tests use a fake embedder and LLM, so they run offline in a few seconds.

## Limitations

- Scanned (image-only) PDFs have no text layer and are rejected; there is no OCR.
- The local embedding model is small and English-focused. Retrieval works well for direct questions but can miss passages phrased very differently from the question; raising **Sources to retrieve** helps.
- One shared password (`APP_PASSWORD`), not per-user accounts: everyone who logs in shares one document collection.
