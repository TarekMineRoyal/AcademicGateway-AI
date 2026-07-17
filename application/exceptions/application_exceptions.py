class ApplicationException(Exception):
    """
    Base domain-agnostic exception for all application-layer errors.
    All custom application exceptions must inherit from this.
    """
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class EmbeddingServiceException(ApplicationException):
    """
    Raised when the embedding generation pipeline fails, times out,
    or returns an unexpected vector dimensional format (e.g., not 768).
    """
    pass


class VectorRepositoryException(ApplicationException):
    """
    Raised when a storage, deletion, or querying operation fails
    within the underlying vector database engine (e.g., LanceDB).
    """
    pass


class EntityNotFoundException(ApplicationException):
    """
    Raised when a query requests matches or data for a primary
    surrogate ID that does not exist within the vector space.
    """
    pass