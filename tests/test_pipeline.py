"""
tests/test_pipeline.py

Suite de tests para el pipeline RAG de MLflow Docs.

Correr todos los tests:
    pytest tests/ -v

Correr solo un grupo:
    pytest tests/ -v -k "CleanMdx"
    pytest tests/ -v -k "Retriever"
    pytest tests/ -v -k "Generator"
    pytest tests/ -v -k "API"
"""

import sys
from unittest.mock import MagicMock, patch

import pytest

from load_and_chunk import (
    chunk_documents,
    clean_mdx,
    extract_title,
    merge_short_pieces,
)


# =============================================================================
# 1. Limpieza y chunking
# =============================================================================

class TestCleanMdx:
    def test_removes_frontmatter(self):
        text = "---\ntitle: Test\ndescription: foo\n---\n\n# Hello\n\nSome content."
        result = clean_mdx(text)
        assert "title: Test" not in result
        assert "# Hello" in result

    def test_removes_jsx_imports(self):
        text = "import Tabs from '@theme/Tabs'\nimport TabItem from '@theme/TabItem'\n\n# Hello"
        result = clean_mdx(text)
        assert "import Tabs" not in result
        assert "# Hello" in result

    def test_removes_jsx_components(self):
        text = "Some text\n<Tabs>\n<TabItem value='py'>content</TabItem>\n</Tabs>\nMore text"
        result = clean_mdx(text)
        assert "<Tabs>" not in result
        assert "<TabItem" not in result
        assert "Some text" in result
        assert "More text" in result

    def test_removes_images(self):
        text = "Some text\n![alt text](/images/example.png)\nMore text"
        result = clean_mdx(text)
        assert "![" not in result
        assert "Some text" in result
        assert "More text" in result

    def test_removes_html_comments(self):
        text = "Before\n<!-- This is a comment -->\nAfter"
        result = clean_mdx(text)
        assert "<!--" not in result
        assert "Before" in result
        assert "After" in result

    def test_collapses_multiple_blank_lines(self):
        text = "Line one\n\n\n\nLine two"
        result = clean_mdx(text)
        assert "\n\n\n" not in result

    def test_preserves_code_blocks(self):
        text = "Example:\n\n```python\nimport mlflow\nmlflow.log_param('alpha', 0.5)\n```"
        result = clean_mdx(text)
        assert "import mlflow" in result
        assert "mlflow.log_param" in result

    def test_preserves_markdown_headings(self):
        text = "# Main Title\n\n## Section\n\nSome content here."
        result = clean_mdx(text)
        assert "# Main Title" in result
        assert "## Section" in result

    def test_empty_file_returns_empty_string(self):
        assert clean_mdx("") == ""

    def test_only_frontmatter_returns_empty(self):
        text = "---\ntitle: Empty\n---\n"
        result = clean_mdx(text)
        assert result == ""


class TestExtractTitle:
    def test_extracts_title_from_frontmatter(self):
        text = "---\ntitle: MLflow Tracking\n---\n\n# Other heading"
        assert extract_title(text, fallback="fallback") == "MLflow Tracking"

    def test_extracts_title_from_h1_when_no_frontmatter(self):
        text = "# MLflow Model Registry\n\nSome content."
        assert extract_title(text, fallback="fallback") == "MLflow Model Registry"

    def test_uses_fallback_when_no_title_found(self):
        text = "Just some text without any heading."
        assert extract_title(text, fallback="my_fallback") == "my_fallback"

    def test_frontmatter_title_takes_priority_over_h1(self):
        text = "---\ntitle: Frontmatter Title\n---\n\n# H1 Title"
        assert extract_title(text, fallback="fallback") == "Frontmatter Title"


class TestMergeShortPieces:
    def test_merges_short_piece_with_next(self):
        pieces = ["## Short header", "This is the content of the section, long enough to pass the threshold."]
        result = merge_short_pieces(pieces, min_chars=80)
        assert len(result) == 1
        assert "## Short header" in result[0]
        assert "long enough" in result[0]

    def test_keeps_long_pieces_separate(self):
        long1 = "A" * 100
        long2 = "B" * 100
        result = merge_short_pieces([long1, long2], min_chars=80)
        assert len(result) == 2

    def test_short_last_piece_merges_into_previous(self):
        pieces = ["A" * 100, "short"]
        result = merge_short_pieces(pieces, min_chars=80)
        assert len(result) == 1
        assert "short" in result[0]

    def test_empty_input_returns_empty(self):
        assert merge_short_pieces([]) == []

    def test_single_short_piece_is_kept(self):
        result = merge_short_pieces(["tiny"], min_chars=80)
        assert len(result) == 1
        assert result[0] == "tiny"


class TestChunkDocuments:
    def test_produces_chunks_from_docs(self):
        docs = [{"source": "test.mdx", "title": "Test", "text": "Word " * 300}]
        chunks = chunk_documents(docs, chunk_size=100, chunk_overlap=10)
        assert len(chunks) > 1

    def test_chunk_has_required_fields(self):
        docs = [{"source": "test.mdx", "title": "Test", "text": "Word " * 200}]
        chunks = chunk_documents(docs)
        for chunk in chunks:
            assert "chunk_id" in chunk
            assert "source" in chunk
            assert "title" in chunk
            assert "text" in chunk

    def test_chunk_id_format(self):
        docs = [{"source": "tracking/index.mdx", "title": "Tracking", "text": "Word " * 200}]
        chunks = chunk_documents(docs)
        assert chunks[0]["chunk_id"].startswith("tracking/index.mdx::")

    def test_source_is_preserved(self):
        docs = [{"source": "model-registry/index.mdx", "title": "Registry", "text": "Word " * 200}]
        chunks = chunk_documents(docs)
        assert all(c["source"] == "model-registry/index.mdx" for c in chunks)

    def test_empty_docs_returns_empty_chunks(self):
        assert chunk_documents([]) == []

    def test_no_chunk_is_empty(self):
        docs = [{"source": "test.mdx", "title": "Test", "text": "Word " * 300}]
        chunks = chunk_documents(docs, chunk_size=100, chunk_overlap=10)
        assert all(len(c["text"].strip()) > 0 for c in chunks)


# =============================================================================
# 2. Retriever
# =============================================================================

class TestRetriever:
    """Mockea QdrantClient y SentenceTransformer para no necesitar servicios externos."""

    def _make_mock_encode(self):
        """Devuelve un mock de encode() que retorna un objeto con .tolist()."""
        mock_vector = MagicMock()
        mock_vector.tolist.return_value = [0.1] * 384
        mock_model = MagicMock()
        mock_model.encode.return_value = mock_vector
        return mock_model

    def _make_mock_point(self):
        mock_point = MagicMock()
        mock_point.score = 0.95
        mock_point.payload = {
            "text": "MLflow tracking content",
            "source": "tracking/index.mdx",
            "title": "Tracking",
        }
        return mock_point

    def test_search_returns_list_of_dicts(self):
        with patch("retriever.QdrantClient") as mock_qdrant_cls, \
             patch("retriever.SentenceTransformer") as mock_st_cls:

            mock_st_cls.return_value = self._make_mock_encode()
            mock_client = MagicMock()
            mock_client.query_points.return_value.points = [self._make_mock_point()]
            mock_qdrant_cls.return_value = mock_client

            from retriever import Retriever
            r = Retriever(qdrant_url="http://localhost:6333", collection="mlflow_docs")
            results = r.search("How does MLflow tracking work?", top_k=1)

            assert isinstance(results, list)
            assert len(results) == 1
            assert results[0]["text"] == "MLflow tracking content"
            assert results[0]["source"] == "tracking/index.mdx"
            assert results[0]["score"] == 0.95

    def test_search_calls_encode_with_query(self):
        with patch("retriever.QdrantClient") as mock_qdrant_cls, \
             patch("retriever.SentenceTransformer") as mock_st_cls:

            mock_model = self._make_mock_encode()
            mock_st_cls.return_value = mock_model
            mock_qdrant_cls.return_value.query_points.return_value.points = []

            from retriever import Retriever
            r = Retriever(qdrant_url="http://localhost:6333", collection="mlflow_docs")
            r.search("test query", top_k=3)

            mock_model.encode.assert_called_once_with("test query")

    def test_search_respects_top_k(self):
        with patch("retriever.QdrantClient") as mock_qdrant_cls, \
             patch("retriever.SentenceTransformer") as mock_st_cls:

            mock_st_cls.return_value = self._make_mock_encode()
            mock_client = MagicMock()
            mock_client.query_points.return_value.points = []
            mock_qdrant_cls.return_value = mock_client

            from retriever import Retriever
            r = Retriever(qdrant_url="http://localhost:6333", collection="mlflow_docs")
            r.search("query", top_k=7)

            call_kwargs = mock_client.query_points.call_args.kwargs
            assert call_kwargs.get("limit") == 7


# =============================================================================
# 3. Generator
# =============================================================================

class TestGenerator:
    def _make_chunks(self):
        return [
            {"text": "MLflow tracking logs parameters.", "source": "tracking/index.mdx", "title": "Tracking"},
            {"text": "Use mlflow.log_param() to log.", "source": "tracking/quickstart/index.mdx", "title": "Quickstart"},
        ]

    def test_build_context_includes_all_sources(self):
        from generator import Generator
        g = Generator()
        context = g.build_context(self._make_chunks())
        assert "tracking/index.mdx" in context
        assert "tracking/quickstart/index.mdx" in context

    def test_build_context_includes_fragment_numbers(self):
        from generator import Generator
        g = Generator()
        context = g.build_context(self._make_chunks())
        assert "Fragmento 1" in context
        assert "Fragmento 2" in context

    def test_build_context_includes_chunk_text(self):
        from generator import Generator
        g = Generator()
        context = g.build_context(self._make_chunks())
        assert "MLflow tracking logs parameters." in context

    def test_answer_returns_string(self):
        mock_ollama = MagicMock()
        mock_ollama.chat.return_value = {"message": {"content": "This is the answer."}}
        sys.modules["ollama"] = mock_ollama

        from generator import Generator
        g = Generator(model="mistral")
        result = g.answer("How do I log params?", self._make_chunks())

        assert isinstance(result, str)
        assert result == "This is the answer."

    def test_answer_calls_ollama_with_correct_model(self):
        mock_ollama = MagicMock()
        mock_ollama.chat.return_value = {"message": {"content": "answer"}}
        sys.modules["ollama"] = mock_ollama

        from generator import Generator
        g = Generator(model="llama3.2:3b")
        g.answer("question", self._make_chunks())

        call_args = mock_ollama.chat.call_args
        model_used = call_args.kwargs.get("model") or (call_args.args[0] if call_args.args else None)
        assert model_used == "llama3.2:3b"


# =============================================================================
# 4. API
# =============================================================================

@pytest.fixture
def api_client():
    """
    Fixture que devuelve un TestClient de FastAPI con Retriever y Generator
    completamente mockeados — no necesita Qdrant ni Ollama arrancados.

    Inyecta los mocks directamente en los atributos del modulo main despues
    de importarlo, evitando el problema de instanciacion a nivel de modulo.
    """
    # Prepara mocks antes de importar main
    with patch("retriever.QdrantClient"), \
         patch("retriever.SentenceTransformer"):

        sys.modules["ollama"] = MagicMock()

        if "main" in sys.modules:
            del sys.modules["main"]

        import main
        from fastapi.testclient import TestClient

        # Inyecta mocks directamente en los objetos del modulo
        mock_retriever = MagicMock()
        mock_retriever.search.return_value = [
            {"text": "MLflow tracks experiments.", "source": "tracking/index.mdx", "title": "Tracking"},
        ]
        mock_generator = MagicMock()
        mock_generator.answer.return_value = "MLflow tracks your experiments."

        main.retriever = mock_retriever
        main.generator = mock_generator

        client = TestClient(main.app)
        yield client, mock_retriever, mock_generator


class TestAPI:
    def test_health_returns_ok(self, api_client):
        client, _, _ = api_client
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_query_returns_200(self, api_client):
        client, _, _ = api_client
        response = client.post("/query", json={"question": "What is MLflow?"})
        assert response.status_code == 200

    def test_query_response_has_answer_and_sources(self, api_client):
        client, _, _ = api_client
        response = client.post("/query", json={"question": "What is MLflow?"})
        data = response.json()
        assert "answer" in data
        assert "sources" in data

    def test_query_answer_is_string(self, api_client):
        client, _, _ = api_client
        response = client.post("/query", json={"question": "What is MLflow?"})
        assert isinstance(response.json()["answer"], str)

    def test_query_sources_is_list(self, api_client):
        client, _, _ = api_client
        response = client.post("/query", json={"question": "What is MLflow?"})
        assert isinstance(response.json()["sources"], list)

    def test_query_sources_are_unique(self, api_client):
        client, _, _ = api_client
        response = client.post("/query", json={"question": "What is MLflow?"})
        sources = response.json()["sources"]
        assert len(sources) == len(set(sources))

    def test_query_passes_question_to_retriever(self, api_client):
        client, mock_retriever, _ = api_client
        client.post("/query", json={"question": "How does autologging work?"})
        call_args = mock_retriever.search.call_args
        assert "How does autologging work?" in str(call_args)

    def test_query_missing_question_returns_422(self, api_client):
        client, _, _ = api_client
        response = client.post("/query", json={})
        assert response.status_code == 422

    def test_query_default_top_k_is_5(self, api_client):
        client, mock_retriever, _ = api_client
        client.post("/query", json={"question": "test"})
        call_args = mock_retriever.search.call_args
        assert "5" in str(call_args)
