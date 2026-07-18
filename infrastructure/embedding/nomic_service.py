from typing import List
from sentence_transformers import SentenceTransformer
from application.interfaces.embedding_service import IEmbeddingService
from infrastructure.config.settings import settings


class NomicEmbeddingService(IEmbeddingService):
    """
    Concrete implementation of the IEmbeddingService interface using SentenceTransformers.
    Runs entirely locally using PyTorch, leveraging CUDA hardware acceleration on the RTX 3050.
    """

    def __init__(self, model_name: str = settings.NOMIC_MODEL_NAME):
        self._model_name = model_name

        # Load the model directly into VRAM on initialization.
        # trust_remote_code=True is required for Nomic's customized BERT architecture.
        self._model = SentenceTransformer(
            model_name,
            trust_remote_code=True,
            device=settings.COMPUTE_DEVICE
        )

    def embed_document(self, text: str) -> List[float]:
        """Generates a local vector embedding optimized for document ingestion."""
        prefixed_text = f"{settings.DOC_PREFIX}{text}"
        # encode returns a NumPy array; convert to a standard Python list
        embeddings = self._model.encode([prefixed_text], normalize_embeddings=True)
        return embeddings[0].tolist()

    def embed_documents_batch(self, texts: List[str]) -> List[List[float]]:
        """Generates a batch of local vector embeddings for high-speed bulk database insertion."""
        if not texts:
            return []

        prefixed_texts = [f"{settings.DOC_PREFIX}{t}" for t in texts]
        embeddings = self._model.encode(prefixed_texts, normalize_embeddings=True)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """Generates a local vector embedding optimized for similarity search queries."""
        prefixed_text = f"{settings.QUERY_PREFIX}{text}"
        embeddings = self._model.encode([prefixed_text], normalize_embeddings=True)
        return embeddings[0].tolist()