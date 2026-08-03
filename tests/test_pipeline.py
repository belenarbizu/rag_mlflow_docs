"""
Ejecutar todos los tests:
pytest tests/test_pipeline -v
"""

import sys
from unittest.mock import MagicMock, patch
import pytest
from load_and_chunk import chunk_documents, clean_mdx, extract_title, merge_short_pieces

class TestCleanMdx:
    def test_removes_frontmatter(self):
        text = "---\ntitle: Test\n---\n\n# Hello\n\nContent."
        assert "title: Test" not in clean_mdx(text)
        assert "# Hello" in clean_mdx(text)

    def test_removes_jsx_imports(self):
        text = "import Tabs from '@theme/Tabs'\n\n# Hello"
        assert "import Tabs" not in clean_mdx(text)
        assert "# Hello" in clean_mdx(text)

    def test_removes_jsx_components(self):
        text = "Text\n<Tabs>\n<TabItem>content</TabItem>\n</Tabs>\nMore"
        result = clean_mdx(text)
        assert "<Tabs>" not in result
        assert "Text" in result

    def test_removes_images(self):
        text = "Text\n![alt](/images/example.png)\nMore"
        assert "![" not in clean_mdx(text)

    def test_removes_html_comments(self):
        text = "Before\n<!-- comment -->\nAfter"
        assert "<!--" not in clean_mdx(text)

    def test_collapses_multiple_blank_lines(self):
        assert "\n\n\n" not in clean_mdx("Line\n\n\n\nLine")

    def test_preserves_code_blocks(self):
        text = "Example:\n\n```python\nimport mlflow\n```"
        assert "import mlflow" in clean_mdx(text)

    def test_preserves_headings(self):
        text = "# Title\n\n## Section\n\nContent."
        result = clean_mdx(text)
        assert "# Title" in result
        assert "## Section" in result

    def test_empty_returns_empty(self):
        assert clean_mdx("") == ""

    def test_only_frontmatter_returns_empty(self):
        assert clean_mdx("---\ntitle: Empty\n---\n") == ""


class TestExtractTitle:
    def test_from_frontmatter(self):
        assert extract_title("---\ntitle: MLflow\n---\n\n# Other", "fb") == "MLflow"

    def test_from_h1(self):
        assert extract_title("# MLflow Registry\n\nContent.", "fb") == "MLflow Registry"

    def test_fallback(self):
        assert extract_title("Just text.", "my_fallback") == "my_fallback"

    def test_frontmatter_priority(self):
        assert extract_title("---\ntitle: FM\n---\n\n# H1", "fb") == "FM"


class TestMergeShortPieces:
    def test_merges_short_with_next(self):
        pieces = ["## Short", "Long enough content to pass the threshold easily here."]
        result = merge_short_pieces(pieces, min_chars=80)
        assert len(result) == 1
        assert "## Short" in result[0]

    def test_keeps_long_pieces_separate(self):
        result = merge_short_pieces(["A" * 100, "B" * 100], min_chars=80)
        assert len(result) == 2

    def test_short_last_merges_into_previous(self):
        result = merge_short_pieces(["A" * 100, "short"], min_chars=80)
        assert len(result) == 1
        assert "short" in result[0]

    def test_empty_returns_empty(self):
        assert merge_short_pieces([]) == []

    def test_single_short_kept(self):
        result = merge_short_pieces(["tiny"], min_chars=80)
        assert result == ["tiny"]


class TestChunkDocuments:
    def test_produces_chunks(self):
        docs = [{"source": "test.mdx", "title": "Test", "text": "Word " * 300}]
        assert len(chunk_documents(docs, chunk_size=100, chunk_overlap=10)) > 1

    def test_chunk_has_required_fields(self):
        docs = [{"source": "test.mdx", "title": "Test", "text": "Word " * 200}]
        for chunk in chunk_documents(docs):
            assert all(k in chunk for k in ["chunk_id", "source", "title", "text"])

    def test_chunk_id_format(self):
        docs = [{"source": "tracking/index.mdx", "title": "T", "text": "Word " * 200}]
        assert chunk_documents(docs)[0]["chunk_id"].startswith("tracking/index.mdx::")

    def test_source_preserved(self):
        docs = [{"source": "registry/index.mdx", "title": "R", "text": "Word " * 200}]
        assert all(c["source"] == "registry/index.mdx" for c in chunk_documents(docs))

    def test_empty_returns_empty(self):
        assert chunk_documents([]) == []

    def test_no_empty_chunks(self):
        docs = [{"source": "test.mdx", "title": "T", "text": "Word " * 300}]
        assert all(len(c["text"].strip()) > 0 for c in chunk_documents(docs, chunk_size=100, chunk_overlap=10))


class TestRetriever:
    def _make_mock_encode(self):
        mock_vector = MagicMock()
        mock_vector.tolist.return_value = [0.1] * 384
        mock_model = MagicMock()
        mock_model.encode.return_value = mock_vector
        return mock_model

    def _make_mock_point(self):
        p = MagicMock()
        p.score = 0.95
        p.payload = {"text": "MLflow content", "source": "tracking/index.mdx", "title": "Tracking"}
        return p

    def test_search_returns_list_of_dicts(self):
        with patch("retriever.QdrantClient") as mock_qdrant_cls, \
             patch("retriever.SentenceTransformer") as mock_st_cls:
            mock_st_cls.return_value = self._make_mock_encode()
            mock_client = MagicMock()
            mock_client.query_points.return_value.points = [self._make_mock_point()]
            mock_qdrant_cls.return_value = mock_client
            from retriever import Retriever
            results = Retriever("http://localhost:6333", "mlflow_docs").search("query", top_k=1)
            assert isinstance(results, list)
            assert results[0]["source"] == "tracking/index.mdx"
            assert results[0]["score"] == 0.95

    def test_search_calls_encode_with_query(self):
        with patch("retriever.QdrantClient") as mock_qdrant_cls, \
             patch("retriever.SentenceTransformer") as mock_st_cls:
            mock_model = self._make_mock_encode()
            mock_st_cls.return_value = mock_model
            mock_qdrant_cls.return_value.query_points.return_value.points = []
            from retriever import Retriever
            Retriever("http://localhost:6333", "mlflow_docs").search("test query")
            mock_model.encode.assert_called_once_with("test query")

    def test_search_respects_top_k(self):
        with patch("retriever.QdrantClient") as mock_qdrant_cls, \
             patch("retriever.SentenceTransformer") as mock_st_cls:
            mock_st_cls.return_value = self._make_mock_encode()
            mock_client = MagicMock()
            mock_client.query_points.return_value.points = []
            mock_qdrant_cls.return_value = mock_client
            from retriever import Retriever
            Retriever("http://localhost:6333", "mlflow_docs").search("query", top_k=7)
            assert mock_client.query_points.call_args.kwargs.get("limit") == 7


class TestGenerator:
    def _make_chunks(self):
        return [
            {"text": "MLflow tracks params.", "source": "tracking/index.mdx", "title": "Tracking"},
            {"text": "Use log_param().", "source": "tracking/quickstart/index.mdx", "title": "Quickstart"},
        ]

    def test_build_context_includes_all_sources(self):
        from generator import Generator
        ctx = Generator().build_context(self._make_chunks())
        assert "tracking/index.mdx" in ctx
        assert "tracking/quickstart/index.mdx" in ctx

    def test_build_context_includes_fragment_numbers(self):
        from generator import Generator
        ctx = Generator().build_context(self._make_chunks())
        assert "Fragmento 1" in ctx
        assert "Fragmento 2" in ctx

    def test_build_context_includes_text(self):
        from generator import Generator
        assert "MLflow tracks params." in Generator().build_context(self._make_chunks())

    def test_answer_returns_string(self):
        mock_ollama = MagicMock()
        mock_client = MagicMock()
        mock_client.chat.return_value = {"message": {"content": "This is the answer."}}
        mock_ollama.Client.return_value = mock_client
        sys.modules["ollama"] = mock_ollama
        from generator import Generator
        result = Generator(model="mistral").answer("question?", self._make_chunks())
        assert isinstance(result, str)
        assert result == "This is the answer."

    def test_answer_uses_correct_model(self):
        mock_ollama = MagicMock()
        mock_client = MagicMock()
        mock_client.chat.return_value = {"message": {"content": "answer"}}
        mock_ollama.Client.return_value = mock_client
        sys.modules["ollama"] = mock_ollama
        from generator import Generator
        Generator(model="llama3.2:3b").answer("question", self._make_chunks())
        call_args = mock_client.chat.call_args
        model_used = call_args.kwargs.get("model") or (call_args.args[0] if call_args.args else None)
        assert model_used == "llama3.2:3b"


@pytest.fixture
def api_client():
    """
    Fixture que inyecta mocks de retriever y generator directamente
    en el modulo main, sin necesitar Qdrant ni Ollama arrancados.
    Parchea tambien el evento startup para que no intente conectar.
    """
    mock_retriever = MagicMock()
    mock_retriever.search.return_value = [
        {"text": "MLflow tracks.", "source": "tracking/index.mdx", "title": "Tracking", "score": 0.9},
    ]
    mock_generator = MagicMock()
    mock_generator.answer.return_value = "MLflow tracks your experiments."
    sys.modules["ollama"] = MagicMock()

    if "main" in sys.modules:
        del sys.modules["main"]

    # Parchea startup antes de importar main para que no conecte a Qdrant
    with patch("main.init_retriever", return_value=mock_retriever) as _,          patch("main.Generator", return_value=mock_generator):

        import main

        # Inyecta mocks directamente en el modulo (por si startup ya corrio)
        main.retriever = mock_retriever
        main.generator = mock_generator

        from fastapi.testclient import TestClient

        with TestClient(main.app) as client:
            # Sobreescribe de nuevo despues del startup
            main.retriever = mock_retriever
            main.generator = mock_generator
            yield client, mock_retriever, mock_generator


class TestAPI:
    def test_health_returns_ok(self, api_client):
        client, _, _ = api_client
        assert client.get("/health").json() == {"status": "ok"}

    def test_query_returns_200(self, api_client):
        client, _, _ = api_client
        assert client.post("/query", json={"question": "What is MLflow?"}).status_code == 200

    def test_query_has_answer_and_sources(self, api_client):
        client, _, _ = api_client
        data = client.post("/query", json={"question": "What is MLflow?"}).json()
        assert "answer" in data
        assert "sources" in data

    def test_query_answer_is_string(self, api_client):
        client, _, _ = api_client
        assert isinstance(client.post("/query", json={"question": "test"}).json()["answer"], str)

    def test_query_sources_is_list(self, api_client):
        client, _, _ = api_client
        assert isinstance(client.post("/query", json={"question": "test"}).json()["sources"], list)

    def test_query_sources_are_unique(self, api_client):
        client, _, _ = api_client
        sources = client.post("/query", json={"question": "test"}).json()["sources"]
        assert len(sources) == len(set(sources))

    def test_query_passes_question_to_retriever(self, api_client):
        client, mock_retriever, _ = api_client
        client.post("/query", json={"question": "How does autologging work?"})
        assert "How does autologging work?" in str(mock_retriever.search.call_args)

    def test_query_missing_question_returns_422(self, api_client):
        client, _, _ = api_client
        assert client.post("/query", json={}).status_code == 422

    def test_query_default_top_k_is_5(self, api_client):
        client, mock_retriever, _ = api_client
        client.post("/query", json={"question": "test"})
        assert "5" in str(mock_retriever.search.call_args)
