import logging
from typing import List
from sentence_transformers import SentenceTransformer

from application.exceptions.application_exceptions import EmbeddingServiceException
from application.interfaces.embedding_service import IEmbeddingService
from infrastructure.config.settings import settings

logger = logging.getLogger(__name__)


class NomicEmbeddingService(IEmbeddingService):
    """
    Concrete implementation of the IEmbeddingService interface using SentenceTransformers.
    Runs entirely locally using PyTorch, leveraging CUDA hardware acceleration on the RTX 3050.
    """

    def __init__(self, model_name: str = settings.NOMIC_MODEL_NAME):
        self._model_name = model_name
        logger.info(
            f"Initializing Nomic embedding model '{self._model_name}' on compute device '{settings.COMPUTE_DEVICE}'"
        )

        try:
            # Load the model directly into VRAM on initialization.
            # trust_remote_code=True is required for Nomic's customized BERT architecture.
            self._model = SentenceTransformer(
                model_name,
                trust_remote_code=True,
                device=settings.COMPUTE_DEVICE
            )
            logger.info(f"Nomic embedding model '{self._model_name}' loaded successfully.")
        except Exception as ex:
            logger.error(f"Failed to initialize embedding model '{self._model_name}': {ex}")
            raise EmbeddingServiceException(f"Embedding model initialization failed: {str(ex)}") from ex

    def embed_document(self, text: str) -> List[float]:
        """Generates a local vector embedding optimized for document ingestion."""
        try:
            prefixed_text = f"{settings.DOC_PREFIX}{text}"
            embeddings = self._model.encode([prefixed_text], normalize_embeddings=True)
            return embeddings[0].tolist()
        except Exception as ex:
            logger.error(f"Error generating document embedding: {ex}")
            raise EmbeddingServiceException(f"Document embedding failed: {str(ex)}") from ex

    def embed_documents_batch(self, texts: List[str]) -> List[List[float]]:
        """Generates a batch of local vector embeddings for high-speed bulk database insertion."""
        if not texts:
            return []

        batch_size = len(texts)
        logger.info(f"Generating embeddings for batch of {batch_size} documents...")

        try:
            prefixed_texts = [f"{settings.DOC_PREFIX}{t}" for t in texts]
            embeddings = self._model.encode(prefixed_texts, normalize_embeddings=True)
            logger.info(f"Successfully generated batch embeddings for {batch_size} documents.")
            return embeddings.tolist()
        except Exception as ex:
            logger.error(f"Error generating batch embeddings (batch size: {batch_size}): {ex}")
            raise EmbeddingServiceException(f"Batch embedding generation failed: {str(ex)}") from ex

    def embed_query(self, text: str) -> List[float]:
        """Generates a local vector embedding optimized for similarity search queries."""
        try:
            prefixed_text = f"{settings.QUERY_PREFIX}{text}"
            embeddings = self._model.encode([prefixed_text], normalize_embeddings=True)
            return embeddings[0].tolist()
        except Exception as ex:
            logger.error(f"Error generating query embedding: {ex}")
            raise EmbeddingServiceException(f"Query embedding failed: {str(ex)}") from ex