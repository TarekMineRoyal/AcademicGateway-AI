from abc import ABC, abstractmethod
from typing import List

class IEmbeddingService(ABC):
    """
    Defines the abstract interface contract for generating vector embeddings.
    This keeps the application layer completely decoupled from model-specific
    prefix tokens (like Nomic's task instructions) and machine learning frameworks.
    """

    @abstractmethod
    def embed_document(self, text: str) -> List[float]:
        """
        Generates a vector embedding optimized for document ingestion and indexing.
        The concrete implementation handles model-specific prefixing natively.

        Args:
            text (str): The clean narrative text block representing a domain entity.

        Returns:
            List[float]: A 768-dimensional floating-point array vector.
        """
        pass

    @abstractmethod
    def embed_documents_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generates a batch of vector embeddings optimized for document ingestion.
        Used for high-speed concurrent lookups during bulk data synchronization.

        Args:
            texts (List[str]): A collection of clean domain text blocks.

        Returns:
            List[List[float]]: A list of 768-dimensional float arrays.
        """
        pass

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """
        Generates a vector embedding optimized for nearest-neighbor similarity searches.
        The concrete implementation handles model-specific prefixing natively.

        Args:
            text (str): The clean natural language representation of a search request.

        Returns:
            List[float]: A 768-dimensional floating-point array query vector.
        """
        pass