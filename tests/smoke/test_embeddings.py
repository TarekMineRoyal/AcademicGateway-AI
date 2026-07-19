import pytest
from unittest.mock import patch

# Component Imports
from infrastructure.embedding.nomic_service import NomicEmbeddingService
from infrastructure.config.settings import settings

pytestmark = pytest.mark.smoke


@pytest.fixture(scope="module")
def real_embedding_service():
    """
    Initializes the actual local model into VRAM on the RTX 3050.
    Scoped to 'module' so the heavy hardware load only happens once for this file.
    """
    return NomicEmbeddingService()


# ============================================================================
# 3. CORE EMBEDDING PORT TASK PREFIX TESTS (Real Hardware Execution)
# ============================================================================

def test_embed_document_hardware_smoke_and_prefix_spy(real_embedding_service):
    """
    Smoke tests the RTX/CUDA hardware layer and verifies the explicit document
    prefix is applied before execution.
    """
    raw_text = "Data Engineering department handles all pipeline orchestration templates."
    expected_prefixed_text = f"{settings.DOC_PREFIX}{raw_text}"

    # Use 'wraps' to spy on the method call while letting the actual GPU execution happen
    with patch.object(real_embedding_service._model, 'encode',
                      wraps=real_embedding_service._model.encode) as spy_encode:
        result = real_embedding_service.embed_document(raw_text)

        # 1. Assert the Spy saw the correct prefix token layout
        spy_encode.assert_called_once_with(
            [expected_prefixed_text],
            normalize_embeddings=True
        )

    # 2. Assert the real hardware returned a valid embedding payload
    assert isinstance(result, list)
    assert len(result) > 0
    assert isinstance(result[0], float)


def test_embed_documents_batch_hardware_smoke_and_prefix_spy(real_embedding_service):
    """
    Smoke tests batch execution through PyTorch and verifies all items in
    the batch array inherit the document prefix and unpack into native Python floats.
    """
    raw_texts = [
        "First project workflow detail.",
        "Second cross-functional capability description."
    ]
    expected_prefixed_texts = [f"{settings.DOC_PREFIX}{t}" for t in raw_texts]

    with patch.object(real_embedding_service._model, 'encode',
                      wraps=real_embedding_service._model.encode) as spy_encode:
        result = real_embedding_service.embed_documents_batch(raw_texts)

        spy_encode.assert_called_once_with(
            expected_prefixed_texts,
            normalize_embeddings=True
        )

    # Deepened Assertions to prevent leaking underlying NumPy types
    assert isinstance(result, list)
    assert len(result) == 2
    assert len(result[0]) == len(result[1])  # Dimensions must match
    assert isinstance(result[0], list)  # Inner container must be a standard list
    assert isinstance(result[0][0], float)  # Explicitly catch and reject numpy.float32 types


def test_embed_documents_batch_handles_empty_input_gracefully(real_embedding_service):
    """Verify empty batches short-circuit without spinning up GPU execution pathways."""
    with patch.object(real_embedding_service._model, 'encode',
                      wraps=real_embedding_service._model.encode) as spy_encode:
        result = real_embedding_service.embed_documents_batch([])

        assert result == []
        spy_encode.assert_not_called()


def test_embed_query_hardware_smoke_and_prefix_spy(real_embedding_service):
    """
    Smoke tests query execution and ensures the distinct query task prefix
    is leveraged rather than the document template.
    """
    raw_query = "Find systems looking for Kubernetes orchestration expertise."
    expected_prefixed_query = f"{settings.QUERY_PREFIX}{raw_query}"

    with patch.object(real_embedding_service._model, 'encode',
                      wraps=real_embedding_service._model.encode) as spy_encode:
        result = real_embedding_service.embed_query(raw_query)

        spy_encode.assert_called_once_with(
            [expected_prefixed_query],
            normalize_embeddings=True
        )

    assert isinstance(result, list)
    assert len(result) > 0