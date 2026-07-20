import logging
import threading
import uuid
from typing import List, Optional

from lancedb.index import BTree

from application.exceptions.application_exceptions import VectorRepositoryException
from application.interfaces.vector_repositories import IProfessorVectorRepository
from domain.models.professor import Professor
from infrastructure.persistence.lancedb_client import lancedb_client
from infrastructure.persistence.schema_registry import ProfessorTableSchema

logger = logging.getLogger(__name__)


class ProfessorVectorRepository(IProfessorVectorRepository):
    """
    Concrete implementation of IProfessorVectorRepository using LanceDB.
    Handles persistence, pre-filtered semantic matches, and deletion for faculty member profiles.
    """

    _repo_lock = threading.Lock()

    def __init__(self, table_name: Optional[str] = None, client=lancedb_client) -> None:
        self._client = client
        self._table_name = table_name or "professors"
        self._table = None  # In-memory handle cache to avoid continuous disk I/O

    def reload_table(self) -> None:
        """
        Invalidates the cached table handle so subsequent calls re-fetch
        the active table connection (e.g., following a Blue/Green swap).
        """
        with self._repo_lock:
            logger.info(f"Reloading table handle for '{self._table_name}'. Invalidation cached handle.")
            self._table = None

    def _get_table(self):
        """
        Lazily ensures the physical table exists and returns the active cached handle
        using a thread-safe double-checked lock pattern.
        """
        if self._table is not None:
            return self._table

        try:
            conn = self._client.get_connection()

            if self._table_name in conn.list_tables().tables:
                self._table = conn.open_table(self._table_name)
                return self._table

            with self._repo_lock:
                # Double-check inside the thread guard block
                if self._table_name in conn.list_tables().tables:
                    self._table = conn.open_table(self._table_name)
                    return self._table

                logger.info(f"Table '{self._table_name}' does not exist. Creating schema and indexes...")
                table = conn.create_table(self._table_name, schema=ProfessorTableSchema)
                table.create_index("id", config=BTree())
                table.create_index("is_accepting_projects", config=BTree())
                table.create_index("department", config=BTree())

                self._table = table
                logger.info(f"Table '{self._table_name}' initialized successfully with BTree indexes.")
                return self._table
        except Exception as ex:
            logger.error(f"Failed to access or create table '{self._table_name}': {ex}")
            raise VectorRepositoryException(f"Error initializing table '{self._table_name}': {str(ex)}") from ex

    def upsert(self, professor: Professor, vector: List[float]) -> None:
        """
        Saves or updates a professor faculty profile with its interest embedding.
        Uses LanceDB's high-speed merge_insert to ensure atomic upsert operations.
        """
        try:
            table = self._get_table()
            logger.info(f"Upserting professor profile ID '{professor.id}' ({professor.full_name})")

            # Translate pure domain model coordinates into the physical layout schema
            db_record = ProfessorTableSchema(
                id=str(professor.id),
                full_name=professor.full_name,
                department=professor.department,
                rank=professor.rank,
                is_accepting_projects=professor.is_accepting_projects,
                research_interest_ids=[str(rid) for rid in professor.research_interest_ids],
                about_me=professor.about_me,
                vector=vector  # type: ignore
            )

            # Execute thread-safe upsert matching the stringified primary identifier
            table.merge_insert(on="id") \
                .when_matched_update_all() \
                .when_not_matched_insert_all() \
                .execute([db_record.model_dump()])

            logger.info(f"Successfully upserted professor ID '{professor.id}'.")
        except Exception as ex:
            logger.error(f"Failed to upsert professor ID '{professor.id}': {ex}")
            raise VectorRepositoryException(f"Upsert failed for professor '{professor.id}': {str(ex)}") from ex

    def bulk_upsert(
            self, professors: List[Professor], vectors: List[List[float]]
    ) -> None:
        """
        Bulk upserts multiple professor records with their corresponding vectors in a single batch.
        """
        if not professors:
            logger.warning("Bulk upsert invoked with an empty list of professors. Skipping execution.")
            return

        batch_count = len(professors)
        logger.info(f"Starting bulk upsert for {batch_count} professor records.")

        try:
            table = self._get_table()

            records = [
                ProfessorTableSchema(
                    id=str(prof.id),
                    full_name=prof.full_name,
                    department=prof.department,
                    rank=prof.rank,
                    is_accepting_projects=prof.is_accepting_projects,
                    research_interest_ids=[str(rid) for rid in prof.research_interest_ids],
                    about_me=prof.about_me,
                    vector=vec,  # type: ignore
                ).model_dump()
                for prof, vec in zip(professors, vectors)
            ]

            table.merge_insert(on="id") \
                .when_matched_update_all() \
                .when_not_matched_insert_all() \
                .execute(records)

            logger.info(f"Successfully completed bulk upsert for {batch_count} professors.")
        except Exception as ex:
            logger.error(f"Failed bulk upsert for batch of size {batch_count}: {ex}")
            raise VectorRepositoryException(f"Bulk upsert failed: {str(ex)}") from ex

    def delete(self, entity_id: uuid.UUID) -> None:
        """
        Removes a professor profile vector record from persistent storage by its unique ID.
        Execution is natively idempotent in LanceDB.
        """
        try:
            table = self._get_table()
            str_id = str(entity_id)
            logger.info(f"Deleting professor vector record with ID '{str_id}'")
            table.delete(f"id = '{str_id}'")
            logger.info(f"Deleted professor record ID '{str_id}' successfully.")
        except Exception as ex:
            logger.error(f"Failed to delete professor record ID '{entity_id}': {ex}")
            raise VectorRepositoryException(f"Deletion failed for professor '{entity_id}': {str(ex)}") from ex

    def find_nearest(
            self,
            vector: List[float],
            filter_expression: Optional[str] = None,
            limit: int = 10
    ) -> List[uuid.UUID]:
        """
        Executes a vector search against professors combining similarity weights
        with scalar payload pre-filtering parameters.
        """
        try:
            table = self._get_table()
            logger.info(
                f"Executing vector search on '{self._table_name}' (limit={limit}, filter='{filter_expression}')"
            )

            # Initialize the base vector query builder sequence
            query = table.search(vector)

            # Apply SQL-style predicate pre-filtering BEFORE setting the slice limit
            if filter_expression:
                query = query.where(filter_expression, prefilter=True)

            # Apply the final limit and execute the query builder pipeline
            results = query.limit(limit).to_pydantic(ProfessorTableSchema)

            found_uuids = [uuid.UUID(row.id) for row in results]
            logger.info(f"Vector search on '{self._table_name}' returned {len(found_uuids)} matching records.")
            return found_uuids
        except Exception as ex:
            logger.error(f"Vector search failed on table '{self._table_name}': {ex}")
            raise VectorRepositoryException(f"Vector search failed on professors: {str(ex)}") from ex