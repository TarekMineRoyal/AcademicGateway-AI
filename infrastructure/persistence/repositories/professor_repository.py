import threading
import uuid
from typing import List, Optional

from lancedb.index import BTree

from application.interfaces.vector_repositories import IProfessorVectorRepository
from domain.models.professor import Professor
from infrastructure.persistence.lancedb_client import lancedb_client
from infrastructure.persistence.schema_registry import ProfessorTableSchema


class ProfessorVectorRepository(IProfessorVectorRepository):
    """
    Concrete implementation of IProfessorVectorRepository using LanceDB.
    Handles persistence and pre-filtered semantic matches for faculty member profiles.
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
            self._table = None

    def _get_table(self):
        """
        Lazily ensures the physical table exists and returns the active cached handle
        using a thread-safe double-checked lock pattern.
        """
        if self._table is not None:
            return self._table

        conn = self._client.get_connection()

        if self._table_name in conn.list_tables().tables:
            self._table = conn.open_table(self._table_name)
            return self._table

        with self._repo_lock:
            # Double-check inside the thread guard block
            if self._table_name in conn.list_tables().tables:
                self._table = conn.open_table(self._table_name)
                return self._table

            table = conn.create_table(self._table_name, schema=ProfessorTableSchema)
            table.create_index("id", config=BTree())
            table.create_index("is_accepting_projects", config=BTree())
            table.create_index("department", config=BTree())

            self._table = table
            return self._table

    def upsert(self, professor: Professor, vector: List[float]) -> None:
        """
        Saves or updates a professor faculty profile with its interest embedding.
        Uses LanceDB's high-speed merge_insert to ensure atomic upsert operations.
        """
        table = self._get_table()

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

    def bulk_upsert(
        self, professors: List[Professor], vectors: List[List[float]]
    ) -> None:
        """
        Bulk upserts multiple professor records with their corresponding vectors in a single batch.
        """
        if not professors:
            return

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
        table = self._get_table()

        # Initialize the base vector query builder sequence
        query = table.search(vector)

        # Apply SQL-style predicate pre-filtering BEFORE setting the slice limit
        if filter_expression:
            query = query.where(filter_expression, prefilter=True)

        # Apply the final limit and execute the query builder pipeline
        results = query.limit(limit).to_pydantic(ProfessorTableSchema)

        # Map string identifiers back into pure tracking domain UUIDs
        return [uuid.UUID(row.id) for row in results]