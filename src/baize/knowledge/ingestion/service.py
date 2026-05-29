"""IngestionService: orchestrates the document ingestion pipeline."""

import hashlib
import json
import logging
import uuid
from typing import Any

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import PointIdsList, PointStruct
from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from baize.knowledge.ingestion.chunker import ChunkPlan, HierarchicalChunker
from baize.knowledge.ingestion.converters import convert_to_markdown
from baize.knowledge.ingestion.embedder import KnowledgeEmbedder
from baize.knowledge.ingestion.exceptions import DocumentAlreadyExistsError
from baize.knowledge.ingestion.parser import MarkdownParser
from baize.knowledge.ingestion.qdrant_client import VectorDimMismatchError, ensure_collection
from baize.knowledge.models import KnowledgeBaseModel, KnowledgeChunkModel, KnowledgeDocumentModel

logger = logging.getLogger(__name__)

_EMBED_BATCH_SIZE = 32


class IngestionService:
    """Orchestrates the 10-step document ingestion pipeline.

    Args:
        session: SQLAlchemy async session for database operations.
        embedder: KnowledgeEmbedder for generating vector embeddings.
        qdrant: AsyncQdrantClient for vector database operations.
    """

    def __init__(
        self,
        session: AsyncSession,
        embedder: KnowledgeEmbedder,
        qdrant: AsyncQdrantClient,
    ) -> None:
        self._session = session
        self._embedder = embedder
        self._qdrant = qdrant

    async def ingest_markdown(
        self,
        *,
        kb_id: uuid.UUID,
        title: str,
        content: str,
        source_type: str = "markdown",
        source_uri: str | None = None,
        doc_metadata: dict[str, Any] | None = None,
    ) -> uuid.UUID:
        """Ingest markdown content into a knowledge base.

        Returns:
            The document ID.

        Raises:
            ValueError: KB not found or not active.
            DocumentAlreadyExistsError: Same content already ingested in this KB.
            VectorDimMismatchError: Embedding dimension does not match KB configuration.
        """
        # Step 1: Validate KB
        kb = await self._session.get(KnowledgeBaseModel, kb_id)
        if kb is None:
            raise ValueError(f"Knowledge base {kb_id} not found")
        if kb.status != "active":
            raise ValueError(
                f"Knowledge base {kb_id} is not active (status={kb.status})"
            )

        # Step 2: Content hash
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

        # Step 3: Duplicate check
        result = await self._session.execute(
            select(KnowledgeDocumentModel).where(
                KnowledgeDocumentModel.kb_id == kb_id,
                KnowledgeDocumentModel.content_hash == content_hash,
                KnowledgeDocumentModel.status == "ingested",
            )
        )
        if result.scalar_one_or_none() is not None:
            raise DocumentAlreadyExistsError(
                f"Document with content_hash={content_hash} already ingested in KB {kb_id}"
            )

        # Step 4: Create document record
        doc_id = uuid.uuid4()
        doc = KnowledgeDocumentModel(
            id=doc_id,
            kb_id=kb_id,
            title=title,
            source_type=source_type,
            source_uri=source_uri,
            content_hash=content_hash,
            doc_metadata=doc_metadata if doc_metadata is not None else {},
            status="processing",
        )
        self._session.add(doc)
        await self._session.commit()

        try:
            # Step 5: Parse
            parsed = MarkdownParser.parse(title, content)

            # Step 6: Chunk
            chunk_plans = HierarchicalChunker().chunk(parsed)

            if not chunk_plans:
                doc.status = "ingested"
                doc.chunk_count = 0
                await self._session.commit()
                return doc_id

            # Step 7: Batch embed
            texts = [cp.content for cp in chunk_plans]
            all_vectors: list[list[float]] = []
            for i in range(0, len(texts), _EMBED_BATCH_SIZE):
                batch = texts[i : i + _EMBED_BATCH_SIZE]
                vectors = await self._embedder.embed_documents(batch)
                all_vectors.extend(vectors)

            # Step 8: Validate dimensions
            if all_vectors and len(all_vectors[0]) != kb.embedding_dim:
                raise VectorDimMismatchError(
                    f"Embedder returned dim={len(all_vectors[0])}, "
                    f"KB expects dim={kb.embedding_dim}"
                )

            # Step 9: Postgres transaction — insert all chunks with local_id→db_id mapping
            local_to_db: dict[str, uuid.UUID] = {}
            chunk_models: list[KnowledgeChunkModel] = []
            for cp in chunk_plans:
                chunk_uuid = uuid.uuid4()
                parent_db_id = (
                    local_to_db.get(cp.parent_local_id)
                    if cp.parent_local_id
                    else None
                )
                chunk = KnowledgeChunkModel(
                    id=chunk_uuid,
                    kb_id=kb_id,
                    doc_id=doc_id,
                    parent_chunk_id=parent_db_id,
                    level=cp.level,
                    section_path=(
                        json.dumps(cp.section_path, ensure_ascii=False)
                        if cp.section_path
                        else None
                    ),
                    content=cp.content,
                )
                self._session.add(chunk)
                local_to_db[cp.local_id] = chunk_uuid
                chunk_models.append(chunk)
            await self._session.flush()
            await self._session.commit()

            # Step 10: Qdrant upsert
            collection_name = f"baize_kb_{kb_id.hex}"
            await ensure_collection(
                collection_name, kb.embedding_dim, client=self._qdrant
            )

            point_id_map: dict[uuid.UUID, uuid.UUID] = {}
            points: list[PointStruct] = []
            for cp, chunk_model, vector in zip(chunk_plans, chunk_models, all_vectors):
                point_id = uuid.uuid4()
                point_id_map[chunk_model.id] = point_id
                points.append(
                    PointStruct(
                        id=str(point_id),
                        vector=vector,
                        payload={
                            "chunk_id": str(chunk_model.id),
                            "doc_id": str(doc_id),
                            "kb_id": str(kb_id),
                            "content": cp.content,
                            "level": cp.level,
                            "section_path": cp.section_path,
                        },
                    )
                )

            await self._qdrant.upsert(
                collection_name=collection_name, points=points
            )

            # Step 11: Backfill qdrant_point_id
            for chunk_model in chunk_models:
                chunk_model.qdrant_point_id = point_id_map[chunk_model.id]

            # Step 12: Mark complete
            doc.status = "ingested"
            doc.chunk_count = len(chunk_models)
            await self._session.commit()

            return doc_id

        except Exception as exc:
            await self._session.rollback()
            try:
                await self._session.execute(
                    update(KnowledgeDocumentModel)
                    .where(KnowledgeDocumentModel.id == doc_id)
                    .values(
                        status="failed",
                        error_message=str(exc)[:2000],
                    )
                )
                await self._session.commit()
            except Exception:
                logger.exception(
                    "Failed to mark document %s as failed", doc_id
                )
            raise

    async def process_pending_document(
        self,
        *,
        doc_id: uuid.UUID,
        kb_id: uuid.UUID,
        content: str,
    ) -> None:
        """Process an already-created pending document through the ingestion pipeline.

        Updates status to 'ingesting', then parses/chunks/embeds/stores the content.
        On success, marks 'ingested' with chunk_count. On failure, marks 'failed' with
        error_message and calls cleanup_failed_document to remove residual chunks.

        Args:
            doc_id: ID of the pending document (already exists in DB).
            kb_id: Knowledge base ID.
            content: Markdown content to process.
        """
        kb = await self._session.get(KnowledgeBaseModel, kb_id)
        if kb is None:
            raise ValueError(f"Knowledge base {kb_id} not found")

        doc = await self._session.get(KnowledgeDocumentModel, doc_id)
        if doc is None:
            raise ValueError(f"Document {doc_id} not found")

        doc.status = "ingesting"
        await self._session.commit()

        try:
            parsed = MarkdownParser.parse(doc.title, content)
            chunk_plans = HierarchicalChunker().chunk(parsed)

            if not chunk_plans:
                doc.status = "ingested"
                doc.chunk_count = 0
                await self._session.commit()
                return

            texts = [cp.content for cp in chunk_plans]
            all_vectors: list[list[float]] = []
            for i in range(0, len(texts), _EMBED_BATCH_SIZE):
                batch = texts[i : i + _EMBED_BATCH_SIZE]
                vectors = await self._embedder.embed_documents(batch)
                all_vectors.extend(vectors)

            if all_vectors and len(all_vectors[0]) != kb.embedding_dim:
                raise VectorDimMismatchError(
                    f"Embedder returned dim={len(all_vectors[0])}, "
                    f"KB expects dim={kb.embedding_dim} (dim mismatch)"
                )

            local_to_db: dict[str, uuid.UUID] = {}
            chunk_models: list[KnowledgeChunkModel] = []
            for cp in chunk_plans:
                chunk_uuid = uuid.uuid4()
                parent_db_id = (
                    local_to_db.get(cp.parent_local_id) if cp.parent_local_id else None
                )
                chunk = KnowledgeChunkModel(
                    id=chunk_uuid,
                    kb_id=kb_id,
                    doc_id=doc_id,
                    parent_chunk_id=parent_db_id,
                    level=cp.level,
                    section_path=(
                        json.dumps(cp.section_path, ensure_ascii=False)
                        if cp.section_path
                        else None
                    ),
                    content=cp.content,
                )
                self._session.add(chunk)
                local_to_db[cp.local_id] = chunk_uuid
                chunk_models.append(chunk)
            await self._session.flush()
            await self._session.commit()

            collection_name = f"baize_kb_{kb_id.hex}"
            await ensure_collection(collection_name, kb.embedding_dim, client=self._qdrant)

            point_id_map: dict[uuid.UUID, uuid.UUID] = {}
            points: list[PointStruct] = []
            for cp, chunk_model, vector in zip(chunk_plans, chunk_models, all_vectors):
                point_id = uuid.uuid4()
                point_id_map[chunk_model.id] = point_id
                points.append(
                    PointStruct(
                        id=str(point_id),
                        vector=vector,
                        payload={
                            "chunk_id": str(chunk_model.id),
                            "doc_id": str(doc_id),
                            "kb_id": str(kb_id),
                            "content": cp.content,
                            "level": cp.level,
                            "section_path": cp.section_path,
                        },
                    )
                )

            await self._qdrant.upsert(collection_name=collection_name, points=points)

            for chunk_model in chunk_models:
                chunk_model.qdrant_point_id = point_id_map[chunk_model.id]

            doc.status = "ingested"
            doc.chunk_count = len(chunk_models)
            await self._session.commit()

        except Exception as exc:
            await self._session.rollback()
            try:
                await self._session.execute(
                    update(KnowledgeDocumentModel)
                    .where(KnowledgeDocumentModel.id == doc_id)
                    .values(status="failed", error_message=str(exc)[:2000])
                )
                await self._session.commit()
                await self.cleanup_failed_document(doc_id=doc_id)
            except Exception:
                logger.exception(
                    "Failed to mark document %s as failed after ingestion error", doc_id
                )
            raise

    async def cleanup_failed_document(self, *, doc_id: uuid.UUID) -> None:
        """Delete all chunks and Qdrant points for a document.

        Does not delete the document record itself. Safe to call when the
        document has no chunks or does not exist.

        Args:
            doc_id: The document ID whose chunks should be cleaned up.
        """
        result = await self._session.execute(
            select(KnowledgeChunkModel).where(KnowledgeChunkModel.doc_id == doc_id)
        )
        chunks = result.scalars().all()

        if not chunks:
            return

        point_ids = [str(c.qdrant_point_id) for c in chunks if c.qdrant_point_id is not None]
        if point_ids:
            kb_id = chunks[0].kb_id
            collection_name = f"baize_kb_{kb_id.hex}"
            await self._qdrant.delete(
                collection_name=collection_name,
                points_selector=PointIdsList(points=point_ids),
            )

        await self._session.execute(
            delete(KnowledgeChunkModel).where(KnowledgeChunkModel.doc_id == doc_id)
        )
        await self._session.commit()

    async def ingest_file(
        self,
        *,
        kb_id: uuid.UUID,
        title: str,
        raw_bytes: bytes,
        source_type: str,
        source_uri: str | None = None,
        doc_metadata: dict[str, Any] | None = None,
    ) -> uuid.UUID:
        """Convert raw bytes to Markdown and ingest into a knowledge base.

        Calls convert_to_markdown to convert the raw bytes, then delegates to
        ingest_markdown for the full ingestion pipeline. Converter warnings are
        stored in doc_metadata["converter_warnings"].

        Returns:
            The document ID.

        Raises:
            UnsupportedFormatError: If source_type is not supported.
            DocumentDecodeError: If the document cannot be decoded (e.g. encrypted PDF).
            ScannedPdfError: If the PDF has no extractable text.
            ValueError: KB not found or not active.
            DocumentAlreadyExistsError: Same content already ingested in this KB.
            VectorDimMismatchError: Embedding dimension does not match KB configuration.
        """
        converted = await convert_to_markdown(source_type=source_type, raw_bytes=raw_bytes)

        meta = dict(doc_metadata) if doc_metadata else {}
        meta["converter_warnings"] = converted.warnings

        return await self.ingest_markdown(
            kb_id=kb_id,
            title=title,
            content=converted.markdown,
            source_type=source_type,
            source_uri=source_uri,
            doc_metadata=meta,
        )
