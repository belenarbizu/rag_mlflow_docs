"""
load_and_chunk.py

Lee todos los .mdx/.md de una carpeta, limpia el contenido (quita
frontmatter YAML y componentes JSX), los divide en chunks y guarda
el resultado en un archivo .jsonl listo para generar embeddings.

Uso:
    python load_and_chunk.py --input ./classic-ml --output ./chunks.jsonl
"""

import argparse
import json
import re
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- Limpieza de MDX -------------------------------------------------

FRONTMATTER_RE = re.compile(r"^---\s*\n.*?\n---\s*\n", re.DOTALL)
IMPORT_RE = re.compile(r"^import .*?$", re.MULTILINE)
JSX_TAG_RE = re.compile(r"</?[A-Z][\w]*[^>]*>")  # Tabs, TabItem, etc (empiezan en mayúscula)
HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
MULTI_BLANK_RE = re.compile(r"\n{3,}")


def clean_mdx(text: str) -> str:
    text = FRONTMATTER_RE.sub("", text)
    text = IMPORT_RE.sub("", text)
    text = JSX_TAG_RE.sub("", text)
    text = HTML_COMMENT_RE.sub("", text)
    text = MULTI_BLANK_RE.sub("\n\n", text)
    return text.strip()


def extract_title(text: str, fallback: str) -> str:
    """Intenta sacar el título del frontmatter o del primer # heading."""
    fm_match = re.search(r"title:\s*(.+)", text)
    if fm_match:
        return fm_match.group(1).strip().strip('"').strip("'")
    h1_match = re.search(r"^#\s+(.+)", text, re.MULTILINE)
    if h1_match:
        return h1_match.group(1).strip()
    return fallback


# --- Carga y chunking --------------------------------------------------

def load_documents(input_dir: Path):
    docs = []
    for path in input_dir.rglob("*"):
        if path.suffix not in (".mdx", ".md"):
            continue
        raw = path.read_text(encoding="utf-8", errors="ignore")
        title = extract_title(raw, fallback=path.stem)
        cleaned = clean_mdx(raw)
        if len(cleaned) < 50:  # se salta archivos vacíos o casi vacíos
            continue
        docs.append({
            "source": str(path.relative_to(input_dir)),
            "title": title,
            "text": cleaned,
        })
    return docs


MIN_CHUNK_CHARS = 80


def merge_short_pieces(pieces, min_chars=MIN_CHUNK_CHARS):
    """Funde un piece muy corto (ej. un header solo) con el siguiente,
    para que ningun chunk quede sin contenido util."""
    merged = []
    buffer = ""
    for piece in pieces:
        buffer = f"{buffer}\n\n{piece}".strip() if buffer else piece
        if len(buffer) >= min_chars:
            merged.append(buffer)
            buffer = ""
    if buffer:  # sobra al final, se pega al ultimo chunk
        if merged:
            merged[-1] = f"{merged[-1]}\n\n{buffer}"
        else:
            merged.append(buffer)
    return merged


def chunk_documents(docs, chunk_size=500, chunk_overlap=75):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " "],
    )
    chunks = []
    for doc in docs:
        pieces = splitter.split_text(doc["text"])
        pieces = merge_short_pieces(pieces)
        for i, piece in enumerate(pieces):
            chunks.append({
                "chunk_id": f"{doc['source']}::{i}",
                "source": doc["source"],
                "title": doc["title"],
                "text": piece,
            })
    return chunks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Carpeta con los .mdx/.md")
    parser.add_argument("--output", required=True, help="Archivo .jsonl de salida")
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--chunk-overlap", type=int, default=75)
    args = parser.parse_args()

    input_dir = Path(args.input)
    docs = load_documents(input_dir)
    print(f"Documentos cargados: {len(docs)}")

    chunks = chunk_documents(docs, args.chunk_size, args.chunk_overlap)
    print(f"Chunks generados: {len(chunks)}")

    with open(args.output, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"Guardado en: {args.output}")


if __name__ == "__main__":
    main()
