import logging
import threading
import uuid
from typing import List, Optional

from lancedb.index import BTree

from application.exceptions.application_exceptions import VectorRepositoryException
from application.interfaces.vector_repositories import ISkillVectorRepository
from domain.models.skill import Skill
from infrastructure.persistence.lancedb_client import lancedb_client
from infrastructure.persistence.schema_registry import SkillTableSchema

logger = logging.getLogger(__name__)


class SkillVectorRepository(ISkillVectorRepository):
    """
    Concrete implementation of ISkillVectorRepository using LanceDB.
    Handles the persistence, semantic queries, batch resolution, and deletion of technical skills.
    """

    _repo_lock = threading.Lock()

    def __init__(self, table_name: Optional[str] = None, client=lancedb_client) -> None:
        self._client = client
        self._table_name = table_name or "skills"
        self._table = None  # In-memory handle cache to avoid continuous disk I/O

    def reload_table(self) -> None:
        """
        Invalidates the cached table handle so subsequent calls re-fetch
        the active table connection (e.g., following a Blue/Green swap).
        """
        with self._repo_lock:
            logger.info(f"Reloading table handle for '{self._table_name}'. Invalidating cached handle.")
            self._table = None

    def _get_table(self):
        if self._table is not None:
            return self._table

        try:
            conn = self._client.get_connection()

            if self._table_name in conn.list_tables().tables:
                self._table = conn.open_table(self._table_name)
                return self._table

            with self._repo_lock:
                if self._table_name in conn.list_tables().tables:
                    self._table = conn.open_table(self._table_name)
                    return self._table

                logger.info(f"Table '{self._table_name}' does not exist. Creating schema and indexes...")
                table = conn.create_table(self._table_name, schema=SkillTableSchema)
                table.create_index("id", config=BTree())

                self._table = table
                logger.info(f"Table '{self._table_name}' initialized successfully with BTree indexes.")
                return self._table
        except Exception as ex:
            logger.error(f"Failed to access or create table '{self._table_name}': {ex}")
            raise VectorRepositoryException(f"Error initializing table '{self._table_name}': {str(ex)}") from ex

    def upsert(self, skill: Skill, vector: List[float]) -> None:
        """
        Saves or updates a standalone skill entity with its corresponding semantic vector.
        Uses LanceDB's high-speed merge_insert to ensure atomic upsert operations.
        """
        try:
            table = self._get_table()
            logger.info(f"Upserting skill record ID '{skill.id}' ({skill.name})")

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

            logger.info(f"Successfully upserted skill ID '{skill.id}'.")
        except Exception as ex:
            logger.error(f"Failed to upsert skill ID '{skill.id}': {ex}")
            raise VectorRepositoryException(f"Upsert failed for skill '{skill.id}': {str(ex)}") from ex

    def bulk_upsert(self, skills: List[Skill], vectors: List[List[float]]) -> None:
        """
        Bulk upserts multiple skill records with their corresponding vectors in a single batch.
        """
        if not skills:
            logger.warning("Bulk upsert invoked with an empty list of skills. Skipping execution.")
            return

        batch_count = len(skills)
        logger.info(f"Starting bulk upsert for {batch_count} skill records.")

        try:
            table = self._get_table()

            records = [
                SkillTableSchema(
                    id=str(sk.id),
                    name=sk.name,
                    vector=vec,  # type: ignore
                ).model_dump()
                for sk, vec in zip(skills, vectors)
            ]

            table.merge_insert(on="id") \
                .when_matched_update_all() \
                .when_not_matched_insert_all() \
                .execute(records)

            logger.info(f"Successfully completed bulk upsert for {batch_count} skills.")
        except Exception as ex:
            logger.error(f"Failed bulk upsert for batch of size {batch_count}: {ex}")
            raise VectorRepositoryException(f"Bulk upsert failed: {str(ex)}") from ex

    def delete(self, entity_id: uuid.UUID) -> None:
        """
        Removes a skill vector record from persistent storage by its unique ID.
        Execution is natively idempotent in LanceDB.
        """
        try:
            table = self._get_table()
            str_id = str(entity_id)
            logger.info(f"Deleting skill vector record with ID '{str_id}'")
            table.delete(f"id = '{str_id}'")
            logger.info(f"Deleted skill record ID '{str_id}' successfully.")
        except Exception as ex:
            logger.error(f"Failed to delete skill record ID '{entity_id}': {ex}")
            raise VectorRepositoryException(f"Deletion failed for skill '{entity_id}': {str(ex)}") from ex

    def find_nearest(
        self,
        vector: List[float],
        filter_expression: Optional[str] = None,
        limit: int = 10
    ) -> List[uuid.UUID]:
        """
        Executes a vector search against the skills catalog to find semantically related capabilities.
        """
        try:
            table = self._get_table()
            logger.info(
                f"Executing vector search on '{self._table_name}' (limit={limit}, filter='{filter_expression}')"
            )

            # Initialize the base vector query builder sequence
            query = table.search(vector)

            # Apply SQL-style predicate pre-filtering BEFORE setting the slice limit if present
            if filter_expression:
                query = query.where(filter_expression, prefilter=True)

            # Apply the final limit and execute the query builder pipeline
            results = query.limit(limit).to_pydantic(SkillTableSchema)

            found_uuids = [uuid.UUID(row.id) for row in results]
            logger.info(f"Vector search on '{self._table_name}' returned {len(found_uuids)} matching records.")
            return found_uuids
        except Exception as ex:
            logger.error(f"Vector search failed on table '{self._table_name}': {ex}")
            raise VectorRepositoryException(f"Vector search failed on skills: {str(ex)}") from ex

    def get_names_by_ids(self, skill_ids: List[uuid.UUID]) -> List[str]:
        """
        Resolves a batch collection of Skill IDs into their raw text string names.
        Optimized to bypass vector column parsing using columnar projection.
        """
        if not skill_ids:
            return []

        try:
            table = self._get_table()
            id_count = len(skill_ids)
            logger.info(f"Resolving names for {id_count} skill IDs...")

            # Format UUIDs into an optimized SQL 'IN' predicate expression
            id_strings = [f"'{str(sid)}'" for sid in skill_ids]
            filter_clause = f"id IN ({', '.join(id_strings)})"

            results = table.search() \
                           .where(filter_clause) \
                           .select(["name"]) \
                           .to_list()

            resolved_names = [row["name"] for row in results]
            logger.info(f"Resolved {len(resolved_names)} names for requested skill IDs.")
            return resolved_names
        except Exception as ex:
            logger.error(f"Failed to resolve skill names by IDs: {ex}")
            raise VectorRepositoryException(f"Failed to resolve skill names: {str(ex)}") from ex