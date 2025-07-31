import sys
import os
import base64
import re
import io
import json
from mistralai import Mistral
from openai import OpenAI
import markdown
from PIL import Image
import shutil
from datetime import datetime
from typing import List, Dict, Any
import hashlib
import faiss
import numpy as np

from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    HTMLHeaderTextSplitter,
    Language
)
from langchain.docstore.document import Document
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS



print("[SCRIPT] Multi-Document Advanced Chunker with Context Management")

def file_is_accepted(file):
    accepted = ['.jpg', '.jpeg', '.png', '.pdf', '.docx', '.txt', '.md']

def count_tokens(text, model="text-embedding-3-small"):
    import tiktoken
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

def create_content_hash(text):
    return hashlib.md5(text.encode('utf-8')).hexdigest()[:8]

def encode_file_base64(filepath):
    print(f"[STEP] Encoding {os.path.basename(filepath)} to base64")
    try:
        with open(filepath, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')
    except Exception as e:
        print(f"[ERROR] Failed to encode file: {e}")
        return None

def extract_headings_from_text(text):
    headings = []
    lines = text.split('\n')
    current_page = 1
    for line_num, line in enumerate(lines):
        line = line.strip()
        if line.startswith('#'):
            level = len(line) - len(line.lstrip('#'))
            heading_text = line.lstrip('#').strip()
            headings.append({
                'level': level,
                'text': heading_text,
                'line_number': line_num + 1,
                'estimated_page': current_page
            })
        if line_num > 0 and line_num % 50 == 0:
            current_page += 1
    return headings

def generate_document_summary(text, openai_api_key, doc_name):
    if not openai_api_key:
        return f"Document: {doc_name}"
    print(f"[STEP] Generating summary for {doc_name}")
    try:
        client = OpenAI(api_key=openai_api_key.strip())
        summary_text = text[:3000] if len(text) > 3000 else text
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": f"Provide a concise 2-3 sentence summary of this document that explains its main topic and content type:\n\n{summary_text}"
                }
            ],
            max_tokens=150
        )
        summary = response.choices[0].message.content.strip()
        print(f"[SUCCESS] Generated summary for {doc_name}")
        return summary
    except Exception as e:
        print(f"[WARNING] Summary generation failed for {doc_name}: {e}")
        return f"Document: {doc_name}"

def advanced_chunker(text, filepath, document_summary, headings, chunk_size=300, chunk_overlap=50, strategy="recursive"):
    print(f"[CHUNKING] Processing {os.path.basename(filepath)} with {strategy} strategy")
    file_ext = os.path.splitext(filepath)[1].lower()
    doc_name = os.path.basename(filepath)
    chunks = []

    if strategy == "markdown" and (file_ext == '.md' or '# ' in text or '## ' in text):
        headers_to_split_on = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3"), ("####", "Header 4")]
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)
        header_chunks = markdown_splitter.split_text(text)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=count_tokens)
        for header_chunk in header_chunks:
            if count_tokens(header_chunk.page_content) > chunk_size:
                sub_chunks = text_splitter.split_text(header_chunk.page_content)
                for sub_chunk in sub_chunks:
                    metadata = header_chunk.metadata.copy()
                    metadata['sub_chunk'] = True
                    chunks.append(Document(page_content=sub_chunk, metadata=metadata))
            else:
                chunks.append(header_chunk)

    elif strategy == "html" and ('<html' in text.lower() or '<div' in text.lower()):
        headers_to_split_on = [("h1", "Header 1"), ("h2", "Header 2"), ("h3", "Header 3"), ("h4", "Header 4")]
        html_splitter = HTMLHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        chunks = html_splitter.split_text(text)

    elif strategy == "code":
        language = Language.PYTHON if file_ext == '.py' else Language.JS if file_ext == '.js' else Language.JAVA if file_ext == '.java' else Language.CPP
        code_splitter = RecursiveCharacterTextSplitter.from_language(language=language, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        split_texts = code_splitter.split_text(text)
        chunks = [Document(page_content=chunk) for chunk in split_texts]

    else:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=count_tokens)
        split_texts = text_splitter.split_text(text)
        chunks = [Document(page_content=chunk) for chunk in split_texts]

    for i, chunk in enumerate(chunks):
        chunk_start = text.find(chunk.page_content[:50])
        relevant_heading = None
        best_distance = float('inf')
        for heading in headings:
            heading_pos = heading['line_number'] * 20
            if heading_pos <= chunk_start:
                distance = chunk_start - heading_pos
                if distance < best_distance:
                    best_distance = distance
                    relevant_heading = heading
        estimated_page = max(1, chunk_start // 2000) if chunk_start >= 0 else 1
        chunk.metadata.update({
            'document_name': doc_name,
            'document_summary': document_summary,
            'source_path': filepath,
            'chunk_id': i,
            'page_number': estimated_page,
            'current_heading': relevant_heading['text'] if relevant_heading else None,
            'heading_level': relevant_heading['level'] if relevant_heading else None,
            'token_count': count_tokens(chunk.page_content),
            'char_count': len(chunk.page_content),
            'content_hash': create_content_hash(chunk.page_content),
            'created_at': datetime.now().isoformat()
        })

    print(f"[SUCCESS] Created {len(chunks)} chunks from {doc_name}")
    return chunks

def embed_chunks(chunks, openai_api_key):
    client = OpenAI(api_key=openai_api_key)
    for chunk in chunks:
        response = client.embeddings.create(model="text-embedding-3-small", input=chunk.page_content)
        chunk.metadata["embedding"] = response.data[0].embedding
    return chunks

def build_faiss_index(chunks):
    embeddings = np.array([chunk.metadata["embedding"] for chunk in chunks]).astype("float32")
    index = faiss.IndexFlatL2(len(embeddings[0]))
    index.add(embeddings)
    return index

def search_similar_chunks(query, index, chunks, openai_api_key, top_k=5):
    client = OpenAI(api_key=openai_api_key)
    query_embedding = client.embeddings.create(model="text-embedding-3-small", input=query).data[0].embedding
    D, I = index.search(np.array([query_embedding]).astype("float32"), top_k)
    return [chunks[i] for i in I[0]]

if __name__ == "main":
    if len(sys.argv) < 3:
        print("Usage: python multi_doc_chunker.py <file> <OPENAI_API_KEY>")
        sys.exit(1)

    filepath = sys.argv[1]
    openai_api_key = sys.argv[2]

    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    doc_name = os.path.basename(filepath)
    headings = extract_headings_from_text(text)
    summary = generate_document_summary(text, openai_api_key, doc_name)
    chunks = advanced_chunker(text, filepath, summary, headings)
    chunks = embed_chunks(chunks, openai_api_key)
    index = build_faiss_index(chunks)

    with open("chunks.json", "w") as f:
        json.dump([{"content": c.page_content, "metadata": c.metadata} for c in chunks], f)
    faiss.write_index(index, "faiss.index")
    print("[DONE] Chunks and FAISS saved.")