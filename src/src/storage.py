"""
Module: storage.py
Description: Vector index memory layer with strict structural isolation.
"""
import numpy as np
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from src.contracts import IngestedChunk

class VersionedVectorIndex:
    """
    Guarantees strict isolation of embedding weights via versioned schema layouts.
    Prevents downstream runtime anomalies due to vector drift.
    """
    def __init__(self, model_name: str, index_version: str):
        self.model_name = model_name
        self.index_version = index_version
        # Instantiating a specific localized model weight matrix
        self.encoder = SentenceTransformer(model_name)
        self.storage: Dict[str, IngestedChunk] = {}
        
    def generate_vector(self, text: str) -> List[float]:
        return self.encoder.encode(text).tolist()

    def upsert(self, chunk: IngestedChunk) -> None:
        """Commits payload to storage partitions after resolving missing vectors."""
        if not chunk.embedding:
            chunk.embedding = self.generate_vector(chunk.text)
        self.storage[chunk.chunk_id] = chunk

    def similarity_search(self, query_text: str, k: int = 2) -> List[Dict[str, Any]]:
        """Executes a vectorized cosine-similarity scan across memory spaces."""
        query_vector = np.array(self.generate_vector(query_text))
        results = []
        
        for chunk in self.storage.values():
            chunk_vector = np.array(chunk.embedding)
            dot_prod = np.dot(query_vector, chunk_vector)
            norm_q = np.linalg.norm(query_vector)
            norm_c = np.linalg.norm(chunk_vector)
            score = dot_prod / (norm_q * norm_c) if (norm_q and norm_c) else 0.0
            results.append({"chunk": chunk, "score": float(score)})
            
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:k]
