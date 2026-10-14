"""
Module: orchestrator.py
Description: The core generation runtime implementing mathematical confidence cutoffs.
"""
from openai import OpenAI
from src.contracts import RetrievalResponse
from src.storage import VersionedVectorIndex

class RAGOrchestrator:
    """Controls context assembly and manages strict LLM response alignment."""
    def __init__(self, vector_index: VersionedVectorIndex, relevance_threshold: float = 0.5):
        self.index = vector_index
        self.relevance_threshold = relevance_threshold
        self.llm_client = OpenAI() 

    def retrieve(self, query: str) -> RetrievalResponse:
        """Acts as a guardrail filtering weak contextual records."""
        raw_matches = self.index.similarity_search(query, k=2)
        valid_chunks = [m["chunk"] for m in raw_matches if m["score"] >= self.relevance_threshold]
        highest_score = raw_matches[0]["score"] if raw_matches else 0.0
        
        return RetrievalResponse(
            query=query,
            context_chunks=valid_chunks,
            passed_threshold=len(valid_chunks) > 0,
            highest_score=highest_score
        )

    def generate_grounded_answer(self, retrieval_output: RetrievalResponse) -> str:
        """Executes a strictly bounded zero-shot context response loop."""
        if not retrieval_output.passed_threshold:
            return "CRITICAL FAULT: Context failed domain verification metrics. Generation blocked."

        context_str = "\n\n".join([f"[URI: {c.metadata.source_url}]: {c.text}" for c in retrieval_output.context_chunks])
        
        system_prompt = (
            "You operate exclusively as a deterministic processor. Rely ONLY on the verified Context. "
            "If the context contains insufficient text metrics to fully construct the assertion, reply with: "
            "'Information not found in documentation.' Do not reference your baseline training weights under any condition."
        )
        
        response = self.llm_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context:\n{context_str}\n\nQuery: {retrieval_output.query}"}
            ],
            temperature=0.0  # Eradicates randomness out of creative weights
        )
        return response.choices.message.content
