# PDF Question-Answering with RAG

A Retrieval-Augmented Generation (RAG) system that answers questions grounded 
in the content of PDF documents. Instead of relying on an LLM's parametric 
knowledge, this pipeline retrieves relevant text chunks from a vector database 
and constrains the model to answer only from that retrieved context — reducing 
hallucination and keeping answers traceable to source text.

## Pipeline

1. **Document ingestion** — Extracts text from PDF files using `pypdf`.
2. **Chunking** — Implements and compares three strategies:
   - Fixed-size word chunking
   - Fixed-size chunking with overlap
   - Semantic chunking (percentile-based breakpoint detection via LangChain's `SemanticChunker`)
3. **Embedding** — Generates 384-dimensional dense embeddings using 
   `sentence-transformers/all-MiniLM-L6-v2`.
4. **Vector storage & retrieval** — Indexes embeddings in Pinecone (with a 
   local ChromaDB variant also included) for cosine-similarity search.
5. **Answer generation** — Retrieves the top-k most relevant chunks and passes 
   them to `Qwen2.5-3B-Instruct` with a constrained prompt, so the model either 
   answers strictly from context or explicitly states the context is insufficient.
6. **API & deployment** — Exposes a `/generate` endpoint via FastAPI, run inside 
   the notebook and tunneled publicly using Cloudflare Tunnel for live testing.

## Tech stack

- **Embeddings:** Sentence-Transformers (all-MiniLM-L6-v2)
- **Vector DB:** Pinecone (ChromaDB variant also implemented)
- **LLM:** Qwen2.5-3B-Instruct via Hugging Face `transformers`
- **Chunking:** LangChain (`SemanticChunker`), NLTK (sentence tokenization)
- **API:** FastAPI + Uvicorn
- **Tunneling:** Cloudflare Tunnel (`cloudflared`)

## Notes

This was built and run as a Colab notebook, with the FastAPI app cell written 
out via `%%writefile` and served in-process using `nest_asyncio` and a 
background thread.
