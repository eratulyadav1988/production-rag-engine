"""
Module: contracts.py
Author: [Your Name] (UK Global Talent Visa Candidate)
Description: Defines strict structural interfaces for high-throughput RAG pipelines.
"""
import time
from typing import List, Optional
from pydantic import BaseModel, Field

class DocumentMetadata(BaseModel):
    """Enforces absolute rules for metadata schemas to block structural drift."""
    source_url: str = Field(..., description="Source origin of the documentation target")
    doc_version: str = Field(..., description="Semantic version string of the asset")
    last_updated: float = Field(default_factory=time.time)
    category: str = Field(..., description="Functional classification tag")

class IngestedChunk(BaseModel):
    """The formal, immutable payload model representing data inside the index cluster."""
    chunk_id: str
    text: str
    metadata: DocumentMetadata
    embedding: Optional[List[float]] = None

class RetrievalResponse(BaseModel):
    """The contractual data envelope routed to downstream Generative Foundation Models."""
    query: str
    context_chunks: List[IngestedChunk]
    passed_threshold: bool
    highest_score: float
