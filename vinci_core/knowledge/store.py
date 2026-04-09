"""
In-process vector store using ChromaDB.
No server needed — persists to disk at ./data/chroma.
Swap for Pinecone/Weaviate when scaling.
"""

import chromadb
from chromadb.utils import embedding_functions
from typing import List
import hashlib

_client = None
_collection = None


def _get_collection():
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(path="./data/chroma")
        ef = embedding_functions.DefaultEmbeddingFunction()  # uses all-MiniLM-L6-v2
        _collection = _client.get_or_create_collection(
            name="ariston_knowledge",
            embedding_function=ef,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def upsert_documents(documents: List[dict]):
    """Add or update documents in the store. Each doc must have 'content' and 'source'."""
    col = _get_collection()
    ids, texts, metadatas = [], [], []
    for doc in documents:
        content = doc.get("content", "")
        if not content:
            continue
        doc_id = hashlib.md5(content[:200].encode()).hexdigest()
        ids.append(doc_id)
        texts.append(content)
        metadatas.append({k: str(v) for k, v in doc.items() if k != "content"})

    if ids:
        col.upsert(ids=ids, documents=texts, metadatas=metadatas)


def query_store(query: str, n_results: int = 5, where: dict = None) -> List[dict]:
    """Semantic search over stored documents."""
    col = _get_collection()
    count = col.count()
    if count == 0:
        return []

    kwargs = {"query_texts": [query], "n_results": min(n_results, count)}
    if where:
        kwargs["where"] = where

    results = col.query(**kwargs)
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]

    return [{"content": d, **m} for d, m in zip(docs, metas)]
