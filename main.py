"""
Execution Entry Point: main.py
Description: Verifies end-to-end telemetry for the architecture demonstration.
"""
from src.storage import VersionedVectorIndex
from src.pipeline import ScalableIngestionPipeline
from src.orchestrator import RAGOrchestrator

def run_pipeline_demo():
    # 1. Initialize our versioned storage baseline
    vector_cluster = VersionedVectorIndex(model_name="all-MiniLM-L6-v2", index_version="v1.0.0")
    
    # 2. Wire up the infrastructure components
    ingestion_service = ScalableIngestionPipeline(vector_index=vector_cluster)
    orchestration_engine = RAGOrchestrator(vector_index=vector_cluster, relevance_threshold=0.45)
    
    # 3. Simulate high-fidelity target document processing
    mock_payload = {
        "id": "sys_doc_99",
        "url": "https://platform.uk",
        "version": "v4.2.1",
        "category": "infrastructure",
        "body": "The UK compliance matrix mandates data sovereignty. All system architectures must enforce strict Pydantic parsing filters to intercept drift prior to LLM compilation. Model indices require version pinning tags."
    }
    ingestion_service.process_and_load(mock_payload)
    
    # 4. Scenario A: Query passes verification contract bounds
    valid_query = "What does the UK compliance matrix mandate?"
    print(f"\n[Executing Query]: '{valid_query}'")
    payload_out_1 = orchestration_engine.retrieve(valid_query)
    print(f"-> Math Similarity Score Match: {payload_out_1.highest_score:.4f}")
    
    # 5. Scenario B: Query falls below threshold cutoff to block hallucinations
    out_of_bounds_query = "What is the capital flight path of Jupiter?"
    print(f"\n[Executing Query]: '{out_of_bounds_query}'")
    payload_out_2 = orchestration_engine.retrieve(out_of_bounds_query)
    print(f"-> Math Similarity Score Match: {payload_out_2.highest_score:.4f}")
    print(f"-> Guardrail Response Output: {orchestration_engine.generate_grounded_answer(payload_out_2)}")

if __name__ == "__main__":
    run_pipeline_demo()
