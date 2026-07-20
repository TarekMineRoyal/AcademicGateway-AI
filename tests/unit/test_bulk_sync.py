import uuid
from unittest.mock import MagicMock, patch

import pytest

from application.commands.bulk_sync_professor import (
    BulkSyncProfessorCommand,
    BulkSyncProfessorCommandHandler,
)
from application.commands.sync_professor import SyncProfessorCommand
from domain.models.professor import Professor


@pytest.fixture
def mock_embedding_service():
    service = MagicMock()
    # Mock batch embedding to return dummy vectors corresponding to document count
    service.embed_documents_batch.side_effect = lambda docs: [[0.1] * 768 for _ in docs]
    return service


def create_sample_sync_professor_command(prof_id: str) -> SyncProfessorCommand:
    professor = Professor(
        id=uuid.UUID(prof_id),
        full_name="Dr. Alan Turing",
        department="Computer Science",
        rank="Full Professor",
        is_accepting_projects=True,
        research_interest_ids=[uuid.uuid4()],
        about_me="Computing pioneer.",
    )
    return SyncProfessorCommand(
        professor=professor,
        interest_areas=["Artificial Intelligence", "Cryptography"],
    )


class TestBulkSyncProfessorCommandHandler:

    @patch("application.commands.bulk_sync_professor.lancedb_client")
    @patch("application.commands.bulk_sync_professor.ProfessorVectorRepository")
    def test_handle_empty_payload_does_nothing(
        self, _mock_repo_cls, mock_lancedb_client, mock_embedding_service
    ):
        handler = BulkSyncProfessorCommandHandler(
            embedding_service=mock_embedding_service
        )
        command = BulkSyncProfessorCommand(items=[])

        handler.handle(command)

        mock_embedding_service.embed_documents_batch.assert_not_called()
        mock_lancedb_client.swap_tables.assert_not_called()

    @patch("application.commands.bulk_sync_professor.settings")
    @patch("application.commands.bulk_sync_professor.lancedb_client")
    @patch("application.commands.bulk_sync_professor.ProfessorVectorRepository")
    def test_handle_chunks_payload_swaps_tables_and_reloads_cache(
        self,
        mock_repo_cls,
        mock_lancedb_client,
        mock_settings,
        mock_embedding_service,
    ):
        # Configure small batch chunk size to force 3 chunks ([2, 2, 1] for 5 items)
        mock_settings.BATCH_CHUNK_SIZE = 2

        # Prepare distinct mock repository instances for staging vs live
        mock_staging_repo = MagicMock()
        mock_live_repo = MagicMock()

        def repo_factory(table_name=None):
            if table_name == "professors_sync":
                return mock_staging_repo
            return mock_live_repo

        mock_repo_cls.side_effect = repo_factory

        handler = BulkSyncProfessorCommandHandler(
            embedding_service=mock_embedding_service,
            staging_table_name="professors_sync",
            live_table_name="professors",
        )

        # Build 5 items
        items = [
            create_sample_sync_professor_command(
                f"00000000-0000-0000-0000-00000000000{i}"
            )
            for i in range(1, 6)
        ]
        command = BulkSyncProfessorCommand(items=items)

        # Act
        handler.handle(command)

        # Assert chunked execution (3 batch iterations)
        assert mock_embedding_service.embed_documents_batch.call_count == 3
        assert mock_staging_repo.bulk_upsert.call_count == 3

        # Assert Blue/Green table swap
        mock_lancedb_client.swap_tables.assert_called_once_with(
            "professors_sync", "professors"
        )

        # Assert cache invalidation on both table handles
        mock_live_repo.reload_table.assert_called_once()
        mock_staging_repo.reload_table.assert_called_once()