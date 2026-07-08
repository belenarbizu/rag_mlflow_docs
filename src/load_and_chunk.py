import re
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter

FRONTMATTER_RE = re.compile(r"^---\s*\n.*?\n---\s*\n", re.DOTALL)
IMPORT_RE = re.compile(r"^import .*?$", re.MULTILINE)
JSX_TAG_RE = re.compile(r"</?[A-Z][\w]*[^>]*>")
HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
MULTI_BLANK_RE = re.compile(r"\n{3,}")
MIN_CHUNK_CHARS = 80

def clean_mdx(text):
    text = FRONTMATTER_RE.sub("", text)
    # Elimina imports JSX solo fuera de bloques de codigo (```)
    lines = text.split("\n")
    in_code_block = False
    cleaned_lines = []
    for line in lines:
        if line.strip().startswith("```"):
            in_code_block = not in_code_block
        if not in_code_block and IMPORT_RE.match(line):
            continue
        cleaned_lines.append(line)
    text = "\n".join(cleaned_lines)
    text = JSX_TAG_RE.sub("", text)
    text = HTML_COMMENT_RE.sub("", text)
    text = IMAGE_RE.sub("", text)
    text = MULTI_BLANK_RE.sub("\n\n", text)
    return text.strip()

def extract_title(text, fallback):
    fm_match = re.search(r"title:\s*(.+)", text)
    if fm_match:
        return fm_match.group(1).strip().strip('"').strip("'")
    h1_match = re.search(r"^#\s+(.+)", text, re.MULTILINE)
    if h1_match:
        return h1_match.group(1).strip()
    return fallback

def merge_short_pieces(pieces, min_chars=MIN_CHUNK_CHARS):
    merged = []
    buffer = ""
    for piece in pieces:
        buffer = f"{buffer}\n\n{piece}".strip() if buffer else piece
        if len(buffer) >= min_chars:
            merged.append(buffer)
            buffer = ""
    if buffer:
        if merged:
            merged[-1] = f"{merged[-1]}\n\n{buffer}"
        else:
            merged.append(buffer)
    return merged

def load_documents(input_dir):
    docs = []
    for path in Path(input_dir).rglob("*"):
        if path.suffix not in (".mdx", ".md"):
            continue
        raw = path.read_text(encoding="utf-8", errors="ignore")
        title = extract_title(raw, fallback=path.stem)
        cleaned = clean_mdx(raw)
        if len(cleaned) < 50:
            continue
        docs.append({"source": str(path.relative_to(input_dir)), "title": title, "text": cleaned})
    return docs

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
