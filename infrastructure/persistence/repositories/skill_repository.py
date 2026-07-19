import uuid
from typing import List, Optional
import threading
import lancedb
from lancedb.index import BTree

from domain.models.skill import Skill
from application.interfaces.vector_repositories import ISkillVectorRepository
from infrastructure.persistence.lancedb_client import lancedb_client
from infrastructure.persistence.schema_registry import SkillTableSchema


class SkillVectorRepository(ISkillVectorRepository):
    """
    Concrete implementation of ISkillVectorRepository using LanceDB.
    Handles the persistence, semantic queries, and batch resolution of technical skills.
    """

    _repo_lock = threading.Lock()

    def __init__(self, client=lancedb_client) -> None:
        self._client = client
        self._table_name = "skills"
        self._table = None  # In-memory handle cache to avoid continuous disk I/O

    def _get_table(self):
        if self._table is not None:
            return self._table

        conn = self._client.get_connection()

        if self._table_name in conn.list_tables().tables:
            self._table = conn.open_table(self._table_name)
            return self._table

        with self._repo_lock:
            if self._table_name in conn.list_tables().tables:
                self._table = conn.open_table(self._table_name)
                return self._table

            table = conn.create_table(self._table_name, schema=SkillTableSchema)
            table.create_index("id", config=BTree())

            self._table = table
            return self._table

    def upsert(self, skill: Skill, vector: List[float]) -> None:
        """
        Saves or updates a standalone skill entity with its corresponding semantic vector.
        Uses LanceDB's high-speed merge_insert to ensure atomic upsert operations.
        """
        table = self._get_table()

        # Translate pure domain model coordinates into the physical layout schema
        db_record = SkillTableSchema(
            id=str(skill.id),
            name=skill.name,
            vector=vector  # type: ignore
        )

        # Execute thread-safe upsert matching the stringified primary identifier
        table.merge_insert(on="id") \
            .when_matched_update_all() \
            .when_not_matched_insert_all() \
            .execute([db_record.model_dump()])

    def find_nearest(
        self,
        vector: List[float],
        filter_expression: Optional[str] = None,
        limit: int = 10
    ) -> List[uuid.UUID]:
        """
        Executes a vector search against the skills catalog to find semantically related capabilities.
        """
        table = self._get_table()

        # Initialize the base vector query builder sequence
        query = table.search(vector)

        # Apply SQL-style predicate pre-filtering BEFORE setting the slice limit if present
        if filter_expression:
            query = query.where(filter_expression, prefilter=True)

        # Apply the final limit and execute the query builder pipeline
        results = query.limit(limit).to_pydantic(SkillTableSchema)

        # Map string identifiers back into pure tracking domain UUIDs
        return [uuid.UUID(row.id) for row in results]

    def get_names_by_ids(self, skill_ids: List[uuid.UUID]) -> List[str]:
        """
        Resolves a batch collection of Skill IDs into their raw text string names.
        Optimized to bypass vector column parsing using columnar projection.
        """
        if not skill_ids:
            return []

        table = self._get_table()

        # Format UUIDs into an optimized SQL 'IN' predicate expression
        id_strings = [f"'{str(sid)}'" for sid in skill_ids]
        filter_clause = f"id IN ({', '.join(id_strings)})"

        # 1. Execute a pure scalar search (no prefilter flag needed here)
        # 2. .select(["name"]) forces LanceDB to completely ignore the heavy vector column on disk
        # 3. .to_list() returns a raw, lightweight list of dictionaries: [{"name": "Python"}, ...]
        results = table.search() \
                       .where(filter_clause) \
                       .select(["name"]) \
                       .to_list()

        return [row["name"] for row in results]