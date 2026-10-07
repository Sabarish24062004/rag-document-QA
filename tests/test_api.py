"""Tests for the HTTP API (app.api.routes) and frontend serving."""

from app.rag.llm import GenerationError
from tests.conftest import make_pdf

PDF = make_pdf(["The warranty period for the device is two years."])


def upload(client, name="manual.pdf", data=PDF):
    return client.post("/api/documents", files={"file": (name, data, "application/pdf")})


def test_health_and_frontend(client):
    assert client.get("/health").json() == {"status": "healthy"}
    page = client.get("/")
    assert page.status_code == 200 and "Document Q&amp;A" in page.text


def test_status(client):
    body = client.get("/api/status").json()
    assert body["llm_enabled"] is True and body["model"] == "fake-model"
    assert body["documents"] == 0


def test_upload_list_ask_delete_flow(client):
    r = upload(client)
    assert r.status_code == 201
    assert r.json()["chunks"] == 1

    assert [d["filename"] for d in client.get("/api/documents").json()] == ["manual.pdf"]

    r = client.post("/api/ask", json={"question": "How long is the warranty?"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"].endswith("[1].")
    assert body["citations"][0] | {"score": 0} == {
        "number": 1,
        "filename": "manual.pdf",
        "page_number": 1,
        "text": "The warranty period for the device is two years.",
        "score": 0,
        "cited": True,
    }

    pdf = client.get("/api/documents/manual.pdf/file")
    assert pdf.status_code == 200 and pdf.headers["content-type"] == "application/pdf"

    assert client.delete("/api/documents/manual.pdf").status_code == 200
    assert client.get("/api/documents").json() == []
    assert client.delete("/api/documents/manual.pdf").status_code == 404


def test_upload_rejects_non_pdf(client):
    r = upload(client, name="notes.txt", data=b"hello")
    assert r.status_code == 400


def test_upload_rejects_corrupt_pdf(client):
    r = upload(client, data=b"not really a pdf")
    assert r.status_code == 400


def test_upload_rejects_pdf_without_text(client):
    r = upload(client, data=make_pdf([""]))
    assert r.status_code == 422
    assert "Scanned" in r.json()["detail"]


def test_ask_validates_question(client):
    assert client.post("/api/ask", json={"question": ""}).status_code == 422
    assert client.post("/api/ask", json={"question": "   "}).status_code == 422
    assert client.post("/api/ask", json={"question": "x", "top_k": 0}).status_code == 422


def test_ask_reports_llm_failure_as_502(client, pipeline):
    upload(client)

    def fail(question, sources):
        raise GenerationError("The language model request failed: invalid key")

    pipeline.generator.generate = fail
    r = client.post("/api/ask", json={"question": "warranty?"})
    assert r.status_code == 502 and "invalid key" in r.json()["detail"]


def test_file_endpoint_unknown_document(client):
    assert client.get("/api/documents/missing.pdf/file").status_code == 404
