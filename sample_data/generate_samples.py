import fitz
from pathlib import Path


def generate_sample_papers():
    out_dir = Path("sample_data")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Paper 1: Neural Retrieval and Adaptive RAG
    doc1 = fitz.open()

    # Page 1
    p1 = doc1.new_page()
    p1.insert_text((50, 60), "Neural Retrieval and Adaptive Hybrid RAG in Scientific QA", fontsize=16, fontname="helv")
    p1.insert_text((50, 85), "Dr. Elena Vance, Dr. Marcus Brody | Institute for Advanced AI Research", fontsize=10, fontname="helv")
    p1.insert_text((50, 120), "ABSTRACT", fontsize=12, fontname="helv")
    abstract_text = (
        "Modern retrieval-augmented generation systems encounter severe hallucination challenges when reasoning over\n"
        "dense academic PDFs. In this paper, we propose an enterprise hybrid retrieval architecture that synergizes\n"
        "ChromaDB dense semantic embeddings with BM25 sparse keyword indices. Furthermore, we incorporate Maximal\n"
        "Marginal Relevance (MMR) re-ranking with a diversity factor of lambda=0.7 to systematically eliminate redundant\n"
        "passages. Empirical evaluations demonstrate a 42% reduction in hallucination rates and an 18% improvement\n"
        "in source-page citation attribution across multi-hop reasoning benchmarks."
    )
    p1.insert_text((50, 140), abstract_text, fontsize=10, fontname="helv")

    p1.insert_text((50, 240), "1 INTRODUCTION", fontsize=12, fontname="helv")
    intro_text = (
        "Dense vector representations excel at semantic similarity but frequently miss domain-specific identifiers,\n"
        "exact mathematical notation, and rare vocabulary in scientific literature. Conversely, BM25 excels at precision\n"
        "keyword matching but fails when queries express lexical paraphrases. By combining dense and sparse retrievers\n"
        "via a convex combination score alpha * S_dense + (1 - alpha) * S_bm25, our system achieves optimal retrieval\n"
        "coverage across diverse query formulations."
    )
    p1.insert_text((50, 260), intro_text, fontsize=10, fontname="helv")

    # Page 2
    p2 = doc1.new_page()
    p2.insert_text((50, 60), "2 METHODOLOGY: HYBRID RETRIEVAL & MMR DIVERSITY", fontsize=12, fontname="helv")
    method_text = (
        "Our ingestion pipeline processes PDF documents using PyMuPDF block detection to preserve layout hierarchy,\n"
        "section headers, and page provenance. Extracted text is segmented into recursive semantic chunks of 1200\n"
        "characters with 200-character overlap. Each chunk retains metadata containing source file, page number,\n"
        "and section title.\n\n"
        "To mitigate context redundancy, we apply Maximal Marginal Relevance (MMR) re-ranking:\n"
        "MMR(d) = argmax [ lambda * Sim(d, Query) - (1 - lambda) * max_{s in S} Sim(d, s) ]\n\n"
        "When duplicate passages from adjoining pages compete for inclusion, the second passage is penalized by its\n"
        "high cross-similarity to the selected passage. This guarantees that top-K retrieved context presents diverse\n"
        "perspectives rather than repetitive snippets."
    )
    p2.insert_text((50, 85), method_text, fontsize=10, fontname="helv")

    # Page 3
    p3 = doc1.new_page()
    p3.insert_text((50, 60), "3 EXPERIMENTS AND VERIFIED ATTRIBUTION", fontsize=12, fontname="helv")
    exp_text = (
        "We evaluated our architecture on 500 complex research queries across 12 academic fields. Grounded prompt\n"
        "synthesis strictly enforces the bracket format [Doc: filename, Page: p, Section: s] on all factual assertions.\n"
        "Answers were streamed using asynchronous FastAPI Server-Sent Events (SSE), achieving a median time-to-first-token\n"
        "of 180ms.\n\n"
        "Citation verification algorithms confirmed that 96.4% of generated assertions were directly corroborated\n"
        "by source-page text snippets. The integration of hybrid scoring with MMR diversity proved pivotal in eliminating\n"
        "hallucinated citations in high-entropy technical responses."
    )
    p3.insert_text((50, 85), exp_text, fontsize=10, fontname="helv")

    p3.insert_text((50, 240), "4 CONCLUSION", fontsize=12, fontname="helv")
    concl_text = (
        "DocuMind establishes a rigorous, production-ready framework for academic conversational intelligence.\n"
        "The combination of ChromaDB dense embeddings, BM25 keyword matching, MMR diversity re-ranking, and citation\n"
        "verification resolves the critical limitations of standard naive RAG systems."
    )
    p3.insert_text((50, 260), concl_text, fontsize=10, fontname="helv")

    doc1.save(str(out_dir / "neural_retrieval_and_adaptive_rag.pdf"))
    doc1.close()

    # Paper 2: Distributed Vector Databases and Indexing
    doc2 = fitz.open()

    p2_1 = doc2.new_page()
    p2_1.insert_text((50, 60), "Scaling Vector Databases for High-Throughput Academic Search", fontsize=16, fontname="helv")
    p2_1.insert_text((50, 85), "Dr. Julian Thorne, Sarah Jenkins | Distributed Systems Laboratory", fontsize=10, fontname="helv")
    p2_1.insert_text((50, 120), "ABSTRACT", fontsize=12, fontname="helv")
    abs2 = (
        "Vector search engines rely on Hierarchical Navigable Small World (HNSW) graphs to achieve sub-linear query\n"
        "latencies across multi-million vector collections. In this work, we analyze ChromaDB persistence layer scaling,\n"
        "in-memory caching dynamics, and inverted index synchronization with lexical BM25 engines. Our findings reveal\n"
        "that persistent SQLite-backed ChromaDB achieves 99th-percentile search latencies under 12ms for collections\n"
        "under 100,000 document vectors."
    )
    p2_1.insert_text((50, 140), abs2, fontsize=10, fontname="helv")

    p2_1.insert_text((50, 240), "1 INTRODUCTION", fontsize=12, fontname="helv")
    intro2 = (
        "Enterprise document search architectures demand both semantic nuance and low latency. ChromaDB implements\n"
        "HNSW graphs with cosine distance metrics. Concurrently, inverted term indexes powered by BM25 provide\n"
        "deterministic token lookups. Coordinating these two systems requires robust lifecycle management and atomic\n"
        "updates during PDF ingestion."
    )
    p2_1.insert_text((50, 260), intro2, fontsize=10, fontname="helv")

    p2_2 = doc2.new_page()
    p2_2.insert_text((50, 60), "2 ARCHITECTURE AND BENCHMARK RESULTS", fontsize=12, fontname="helv")
    bench2 = (
        "Our benchmarks evaluated ChromaDB indexing throughput on multi-core workstations. Ingestion speed reached\n"
        "450 chunks per second using batched upserts of 200 items. Memory footprint remained under 250MB during steady\n"
        "state querying.\n\n"
        "Lexical BM25 synchronization using serialized inverted indices enabled fast cold-start recovery, loading in\n"
        "under 35ms from persistent disk storage. When deployed behind asynchronous FastAPI services with worker pooling,\n"
        "the system sustained 250 concurrent query streams without request degradation."
    )
    p2_2.insert_text((50, 85), bench2, fontsize=10, fontname="helv")

    doc2.save(str(out_dir / "distributed_vector_databases_and_indexing.pdf"))
    doc2.close()

    print("Sample academic research papers generated successfully in sample_data/")


if __name__ == "__main__":
    generate_sample_papers()
