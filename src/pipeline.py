"""
Module: pipeline.py
Description: Highly predictable text-slicing ingestion engine.
"""
import time
from typing import Dict, Any, List
from pydantic import ValidationError
from src.contracts import DocumentMetadata, IngestedChunk
from src.storage import VersionedVectorIndex

class ScalableIngestionPipeline:
    """Ingestion component that segments documents and embeds tracking keys."""
    def __init__(self, vector_index: VersionedVectorIndex):
        self.index = vector_index

    def segment_text(self, text: str, chunk_size: int = 200) -> List[str]:
        """Sliding window matrix logic to prevent truncation of context tokens."""
        words = text.split()
        return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

    def process_and_load(self, raw_doc: Dict[str, Any]) -> None:
        """Transforms unverified payloads into enterprise-compliant vectorized chunks."""
        try:
            metadata = DocumentMetadata(
                source_url=raw_doc["url"],
                doc_version=raw_doc["version"],
                last_updated=time.time(),
                category=raw_doc["category"]
            )
            
            text_splits = self.segment_text(raw_doc["body"])
            for idx, text_slice in enumerate(text_splits):
                chunk = IngestedChunk(
                    chunk_id=f"{raw_doc['id']}_chunk_{idx}",
                    text=text_slice,
                    metadata=metadata
                )
                self.index.upsert(chunk)
        except ValidationError as error:
            # Demonstrates runtime exception tracking architectures
            print(f"[INGESTION_ERROR] Structural validation failure blocked: {error}")
