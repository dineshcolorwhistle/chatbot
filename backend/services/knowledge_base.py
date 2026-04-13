"""
Knowledge Base Service — 📚 RAG Pipeline Core

Handles the full Retrieval-Augmented Generation pipeline:
  - PDF document loading and text extraction
  - Text chunking with overlap for context preservation
  - Embedding generation via Ollama (nomic-embed-text)
  - Pinecone vector storage (upsert and query)
  - Semantic search for relevant context retrieval

Design:
  - Stateless service — all state lives in Pinecone
  - Uses Ollama for local embeddings (no cloud API cost)
  - Configurable chunk size and overlap
  - Returns ranked context snippets for LLM augmentation
"""

import hashlib
import logging
import os
import re
from dataclasses import dataclass, field

import httpx
from pinecone import Pinecone, ServerlessSpec

from config import pinecone_config, embedding_config

logger = logging.getLogger(__name__)


# ============================================
# Data Types
# ============================================

@dataclass
class TextChunk:
    """A chunk of text extracted from a document.

    Attributes:
        text: The chunk content.
        source: Original document filename.
        chunk_index: Position of this chunk within the document.
        metadata: Additional metadata (page number, section, etc.).
    """

    text: str
    source: str
    chunk_index: int
    metadata: dict = field(default_factory=dict)

    @property
    def chunk_id(self) -> str:
        """Generate a deterministic unique ID for this chunk."""
        content_hash = hashlib.md5(
            f"{self.source}:{self.chunk_index}:{self.text[:100]}".encode()
        ).hexdigest()[:12]
        return f"{self.source.replace(' ', '_')}__chunk_{self.chunk_index}__{content_hash}"


@dataclass
class RetrievalResult:
    """A single result from semantic search.

    Attributes:
        text: The matched text chunk.
        source: Document the chunk came from.
        score: Similarity score (0.0 to 1.0).
    """

    text: str
    source: str
    score: float


# ============================================
# Knowledge Base Service
# ============================================

class KnowledgeBase:
    """RAG pipeline service for document-based knowledge retrieval.

    Manages the full lifecycle:
      1. Extract text from PDFs
      2. Chunk text into manageable pieces
      3. Generate embeddings via Ollama
      4. Store vectors in Pinecone
      5. Query for relevant context

    Attributes:
        _pc: Pinecone client instance.
        _index: Pinecone index reference.
        _embedding_model: Ollama model for embeddings.
        _embedding_url: Ollama embedding API endpoint.
        _chunk_size: Target characters per chunk.
        _chunk_overlap: Overlap between consecutive chunks.
    """

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 200,
    ) -> None:
        """Initialize the Knowledge Base service.

        Args:
            chunk_size: Target number of characters per text chunk.
            chunk_overlap: Number of overlapping characters between chunks.
        """
        self._embedding_model = embedding_config.model
        self._embedding_url = f"{embedding_config.ollama_base_url}/api/embed"
        self._embedding_dimension = embedding_config.dimension
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

        # Initialize Pinecone
        self._pc = Pinecone(api_key=pinecone_config.api_key)
        self._index = None

        logger.info(
            "KnowledgeBase initialized (model: %s, chunk_size: %d, overlap: %d)",
            self._embedding_model,
            self._chunk_size,
            self._chunk_overlap,
        )

    async def initialize(self) -> bool:
        """Create or connect to the Pinecone index.

        Creates the index if it doesn't exist, then connects to it.

        Returns:
            True if the index is ready, False on failure.
        """
        index_name = pinecone_config.index_name

        try:
            # Check if index already exists
            existing_indexes = [
                idx.name for idx in self._pc.list_indexes()
            ]

            if index_name not in existing_indexes:
                logger.info("Creating Pinecone index: %s", index_name)
                self._pc.create_index(
                    name=index_name,
                    dimension=self._embedding_dimension,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud=pinecone_config.cloud,
                        region=pinecone_config.region,
                    ),
                )
                logger.info("Pinecone index '%s' created successfully", index_name)
            else:
                logger.info("Pinecone index '%s' already exists", index_name)

            # Connect to the index
            self._index = self._pc.Index(index_name)

            # Verify connection
            stats = self._index.describe_index_stats()
            logger.info(
                "Connected to Pinecone index '%s' — total vectors: %d",
                index_name,
                stats.total_vector_count,
            )
            return True

        except Exception as e:
            logger.error("Failed to initialize Pinecone index: %s", e)
            return False

    # ============================================
    # Document Ingestion Pipeline
    # ============================================

    async def ingest_documents(self, documents_dir: str) -> dict:
        """Ingest all PDF documents from a directory into Pinecone.

        Full pipeline: Extract → Chunk → Embed → Upsert.

        Args:
            documents_dir: Absolute path to the directory containing PDFs.

        Returns:
            Dictionary with ingestion statistics.
        """
        if not self._index:
            raise RuntimeError("KnowledgeBase not initialized. Call initialize() first.")

        stats = {
            "files_processed": 0,
            "total_chunks": 0,
            "total_vectors_upserted": 0,
            "errors": [],
        }

        # Find all PDF files
        pdf_files = [
            f for f in os.listdir(documents_dir)
            if f.lower().endswith(".pdf")
        ]

        if not pdf_files:
            logger.warning("No PDF files found in: %s", documents_dir)
            stats["errors"].append(f"No PDF files found in {documents_dir}")
            return stats

        logger.info("Found %d PDF files to ingest", len(pdf_files))

        for pdf_file in pdf_files:
            pdf_path = os.path.join(documents_dir, pdf_file)
            try:
                # Step 1: Extract text from PDF
                logger.info("Extracting text from: %s", pdf_file)
                raw_text = self._extract_text_from_pdf(pdf_path)

                if not raw_text.strip():
                    logger.warning("No text extracted from: %s", pdf_file)
                    stats["errors"].append(f"No text extracted from {pdf_file}")
                    continue

                logger.info(
                    "Extracted %d characters from %s",
                    len(raw_text),
                    pdf_file,
                )

                # Step 2: Chunk the text
                chunks = self._chunk_text(raw_text, source=pdf_file)
                logger.info("Created %d chunks from %s", len(chunks), pdf_file)

                # Step 3: Generate embeddings and upsert in batches
                batch_size = 50
                for i in range(0, len(chunks), batch_size):
                    batch = chunks[i:i + batch_size]
                    vectors = await self._embed_and_prepare_vectors(batch)

                    if vectors:
                        self._index.upsert(vectors=vectors)
                        stats["total_vectors_upserted"] += len(vectors)
                        logger.info(
                            "Upserted %d vectors (batch %d/%d) for %s",
                            len(vectors),
                            (i // batch_size) + 1,
                            (len(chunks) + batch_size - 1) // batch_size,
                            pdf_file,
                        )

                stats["files_processed"] += 1
                stats["total_chunks"] += len(chunks)

            except Exception as e:
                logger.error("Error processing %s: %s", pdf_file, e, exc_info=True)
                stats["errors"].append(f"Error processing {pdf_file}: {str(e)}")

        logger.info(
            "Ingestion complete — files: %d, chunks: %d, vectors: %d",
            stats["files_processed"],
            stats["total_chunks"],
            stats["total_vectors_upserted"],
        )

        return stats

    # ============================================
    # Semantic Search / Retrieval
    # ============================================

    async def query(
        self,
        question: str,
        top_k: int = 5,
        score_threshold: float = 0.3,
    ) -> list[RetrievalResult]:
        """Query the knowledge base for relevant context.

        Embeds the question and performs similarity search in Pinecone.

        Args:
            question: The user's question to search for.
            top_k: Maximum number of results to return.
            score_threshold: Minimum similarity score to include.

        Returns:
            List of RetrievalResult sorted by relevance (highest first).
        """
        if not self._index:
            logger.warning("KnowledgeBase not initialized — skipping retrieval")
            return []

        try:
            # Generate embedding for the question
            query_embedding = await self._generate_embedding(question)

            if not query_embedding:
                logger.warning("Failed to generate query embedding")
                return []

            # Search Pinecone
            results = self._index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
            )

            # Filter and format results
            retrieval_results: list[RetrievalResult] = []
            for match in results.get("matches", []):
                score = match.get("score", 0.0)
                if score >= score_threshold:
                    metadata = match.get("metadata", {})
                    retrieval_results.append(
                        RetrievalResult(
                            text=metadata.get("text", ""),
                            source=metadata.get("source", "unknown"),
                            score=score,
                        )
                    )

            logger.info(
                "Knowledge query returned %d results (query: '%s')",
                len(retrieval_results),
                question[:80],
            )

            return retrieval_results

        except Exception as e:
            logger.error("Knowledge base query failed: %s", e)
            return []

    def format_context_for_llm(self, results: list[RetrievalResult]) -> str:
        """Format retrieval results into a context string for the LLM.

        Builds a clean, structured context section that can be injected
        into the system prompt.

        Args:
            results: List of RetrievalResult from a query.

        Returns:
            Formatted context string, or empty string if no results.
        """
        if not results:
            return ""

        context_parts: list[str] = [
            "\n--- COMPANY KNOWLEDGE BASE (use this to answer questions about ColorWhistle) ---"
        ]

        for i, result in enumerate(results, 1):
            context_parts.append(
                f"\n[Source: {result.source} | Relevance: {result.score:.2f}]\n"
                f"{result.text}"
            )

        context_parts.append(
            "\n--- END OF KNOWLEDGE BASE ---\n"
            "IMPORTANT: When the user asks questions about the company, services, "
            "pricing, or related topics, USE the above knowledge base information "
            "to provide accurate answers. Do not make up information — if the answer "
            "is not in the knowledge base, say you're not sure and suggest they "
            "contact the team directly."
        )

        return "\n".join(context_parts)

    # ============================================
    # PDF Text Extraction
    # ============================================

    def _extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text content from a PDF file.

        Uses PyMuPDF (fitz) for reliable text extraction.

        Args:
            pdf_path: Absolute path to the PDF file.

        Returns:
            Extracted text as a single string.
        """
        import fitz  # PyMuPDF

        text_parts: list[str] = []

        with fitz.open(pdf_path) as doc:
            for page_num, page in enumerate(doc, 1):
                page_text = page.get_text("text")
                if page_text.strip():
                    text_parts.append(page_text)

        full_text = "\n".join(text_parts)

        # Clean up the text
        full_text = self._clean_text(full_text)

        return full_text

    def _clean_text(self, text: str) -> str:
        """Clean extracted text — remove artifacts and normalize whitespace.

        Args:
            text: Raw extracted text.

        Returns:
            Cleaned text.
        """
        # Remove excessive whitespace
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Remove page numbers and headers/footers patterns
        text = re.sub(r"\n\d+\s*\n", "\n", text)
        # Normalize spaces
        text = re.sub(r"[ \t]+", " ", text)
        # Remove leading/trailing whitespace per line
        lines = [line.strip() for line in text.split("\n")]
        text = "\n".join(lines)

        return text.strip()

    # ============================================
    # Text Chunking
    # ============================================

    def _chunk_text(self, text: str, source: str) -> list[TextChunk]:
        """Split text into overlapping chunks for embedding.

        Uses a sentence-aware chunking strategy to avoid splitting
        mid-sentence when possible.

        Args:
            text: The full document text.
            source: The source filename for metadata.

        Returns:
            List of TextChunk objects.
        """
        # Split text into sentences
        sentences = re.split(r"(?<=[.!?])\s+", text)

        chunks: list[TextChunk] = []
        current_chunk: list[str] = []
        current_length = 0
        chunk_index = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            sentence_length = len(sentence)

            # If adding this sentence exceeds chunk_size, finalize current chunk
            if current_length + sentence_length > self._chunk_size and current_chunk:
                chunk_text = " ".join(current_chunk)
                chunks.append(
                    TextChunk(
                        text=chunk_text,
                        source=source,
                        chunk_index=chunk_index,
                        metadata={"char_count": len(chunk_text)},
                    )
                )
                chunk_index += 1

                # Keep overlap — take the last few sentences for context
                overlap_chars = 0
                overlap_sentences: list[str] = []
                for s in reversed(current_chunk):
                    overlap_chars += len(s)
                    overlap_sentences.insert(0, s)
                    if overlap_chars >= self._chunk_overlap:
                        break

                current_chunk = overlap_sentences
                current_length = sum(len(s) for s in current_chunk)

            current_chunk.append(sentence)
            current_length += sentence_length

        # Add the final chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append(
                TextChunk(
                    text=chunk_text,
                    source=source,
                    chunk_index=chunk_index,
                    metadata={"char_count": len(chunk_text)},
                )
            )

        return chunks

    # ============================================
    # Embedding Generation
    # ============================================

    async def _generate_embedding(self, text: str) -> list[float] | None:
        """Generate an embedding vector for a single text using Ollama.

        Args:
            text: The text to embed.

        Returns:
            List of floats (the embedding vector), or None on failure.
        """
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    self._embedding_url,
                    json={
                        "model": self._embedding_model,
                        "input": text,
                    },
                )

                if response.status_code != 200:
                    logger.error(
                        "Ollama embedding API error (status %d): %s",
                        response.status_code,
                        response.text[:200],
                    )
                    return None

                data = response.json()
                embeddings = data.get("embeddings", [])

                if embeddings and len(embeddings) > 0:
                    return embeddings[0]

                logger.warning("Ollama returned empty embeddings")
                return None

        except httpx.ConnectError:
            logger.error(
                "Cannot connect to Ollama at %s for embeddings. Is Ollama running?",
                embedding_config.ollama_base_url,
            )
            return None
        except Exception as e:
            logger.error("Embedding generation failed: %s", e)
            return None

    async def _embed_and_prepare_vectors(
        self, chunks: list[TextChunk]
    ) -> list[dict]:
        """Generate embeddings for a batch of chunks and prepare Pinecone vectors.

        Args:
            chunks: List of TextChunk objects to embed.

        Returns:
            List of Pinecone vector dicts ready for upsert.
        """
        vectors: list[dict] = []

        for chunk in chunks:
            embedding = await self._generate_embedding(chunk.text)

            if embedding:
                vectors.append({
                    "id": chunk.chunk_id,
                    "values": embedding,
                    "metadata": {
                        "text": chunk.text,
                        "source": chunk.source,
                        "chunk_index": chunk.chunk_index,
                        **chunk.metadata,
                    },
                })
            else:
                logger.warning(
                    "Skipping chunk %s — embedding failed",
                    chunk.chunk_id,
                )

        return vectors

    # ============================================
    # Index Management
    # ============================================

    async def get_index_stats(self) -> dict:
        """Get statistics about the Pinecone index.

        Returns:
            Dictionary with index statistics.
        """
        if not self._index:
            return {"error": "Index not initialized"}

        try:
            stats = self._index.describe_index_stats()
            return {
                "index_name": pinecone_config.index_name,
                "total_vectors": stats.total_vector_count,
                "dimension": stats.dimension,
                "namespaces": dict(stats.namespaces) if stats.namespaces else {},
            }
        except Exception as e:
            logger.error("Failed to get index stats: %s", e)
            return {"error": str(e)}

    async def clear_index(self) -> bool:
        """Delete all vectors from the Pinecone index.

        Returns:
            True if successful, False otherwise.
        """
        if not self._index:
            return False

        try:
            self._index.delete(delete_all=True)
            logger.info("All vectors deleted from index '%s'", pinecone_config.index_name)
            return True
        except Exception as e:
            logger.error("Failed to clear index: %s", e)
            return False
