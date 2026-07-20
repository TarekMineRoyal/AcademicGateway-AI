import logging
import threading
import uuid
from typing import List, Optional

from lancedb.index import BTree

from application.exceptions.application_exceptions import VectorRepositoryException
from application.interfaces.vector_repositories import IProjectTemplateVectorRepository
from domain.models.project_template import ProjectTemplate
from infrastructure.persistence.lancedb_client import lancedb_client
from infrastructure.persistence.schema_registry import ProjectTemplateTableSchema

logger = logging.getLogger(__name__)


class ProjectTemplateVectorRepository(IProjectTemplateVectorRepository):
    """
    Concrete implementation of IProjectTemplateVectorRepository using LanceDB.
    Handles the mapping, persistence, semantic searches, and deletion for project blueprints.
    """

    _repo_lock = threading.Lock()

    def __init__(self, table_name: Optional[str] = None, client=lancedb_client) -> None:
        self._client = client
        self._table_name = table_name or "project_templates"
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
                table = conn.create_table(self._table_name, schema=ProjectTemplateTableSchema)
                table.create_index("id", config=BTree())
                table.create_index("major_id", config=BTree())
                table.create_index("specialty_id", config=BTree())

                self._table = table
                logger.info(f"Table '{self._table_name}' initialized successfully with BTree indexes.")
                return self._table
        except Exception as ex:
            logger.error(f"Failed to access or create table '{self._table_name}': {ex}")
            raise VectorRepositoryException(f"Error initializing table '{self._table_name}': {str(ex)}") from ex

    def upsert(self, template: ProjectTemplate, vector: List[float]) -> None:
        """
        Saves or updates a project template blueprint with its unified descriptive embedding.
        Uses LanceDB's high-speed merge_insert to ensure atomic upsert operations.
        """
        try:
            table = self._get_table()
            logger.info(f"Upserting project template ID '{template.id}' ({template.title})")

            # Translate pure domain model coordinates into the physical layout schema
            db_record = ProjectTemplateTableSchema(
                id=str(template.id),
                title=template.title,
                description=template.description,
                provider_id=str(template.provider_id),
                created_at=template.created_at,
                skill_ids=[str(sid) for sid in template.skill_ids],
                major_id=str(template.major_id) if template.major_id else None,
                specialty_id=str(template.specialty_id) if template.specialty_id else None,
                vector=vector  # type: ignore
            )

            # Execute thread-safe upsert matching the stringified primary identifier
            table.merge_insert(on="id") \
                .when_matched_update_all() \
                .when_not_matched_insert_all() \
                .execute([db_record.model_dump()])

            logger.info(f"Successfully upserted project template ID '{template.id}'.")
        except Exception as ex:
            logger.error(f"Failed to upsert project template ID '{template.id}': {ex}")
            raise VectorRepositoryException(f"Upsert failed for project template '{template.id}': {str(ex)}") from ex

    def bulk_upsert(
        self, templates: List[ProjectTemplate], vectors: List[List[float]]
    ) -> None:
        """
        Bulk upserts multiple project templates with their corresponding vectors in a single batch.
        """
        if not templates:
            logger.warning("Bulk upsert invoked with an empty list of project templates. Skipping execution.")
            return

        batch_count = len(templates)
        logger.info(f"Starting bulk upsert for {batch_count} project template records.")

        try:
            table = self._get_table()

            records = [
                ProjectTemplateTableSchema(
                    id=str(tmpl.id),
                    title=tmpl.title,
                    description=tmpl.description,
                    provider_id=str(tmpl.provider_id),
                    created_at=tmpl.created_at,
                    skill_ids=[str(sid) for sid in tmpl.skill_ids],
                    major_id=str(tmpl.major_id) if tmpl.major_id else None,
                    specialty_id=str(tmpl.specialty_id) if tmpl.specialty_id else None,
                    vector=vec,  # type: ignore
                ).model_dump()
                for tmpl, vec in zip(templates, vectors)
            ]

            table.merge_insert(on="id") \
                .when_matched_update_all() \
                .when_not_matched_insert_all() \
                .execute(records)

            logger.info(f"Successfully completed bulk upsert for {batch_count} project templates.")
        except Exception as ex:
            logger.error(f"Failed bulk upsert for batch of size {batch_count}: {ex}")
            raise VectorRepositoryException(f"Bulk upsert failed: {str(ex)}") from ex

    def delete(self, entity_id: uuid.UUID) -> None:
        """
        Removes a project blueprint vector record from persistent storage by its unique ID.
        Execution is natively idempotent in LanceDB.
        """
        try:
            table = self._get_table()
            str_id = str(entity_id)
            logger.info(f"Deleting project template vector record with ID '{str_id}'")
            table.delete(f"id = '{str_id}'")
            logger.info(f"Deleted project template record ID '{str_id}' successfully.")
        except Exception as ex:
            logger.error(f"Failed to delete project template record ID '{entity_id}': {ex}")
            raise VectorRepositoryException(f"Deletion failed for project template '{entity_id}': {str(ex)}") from ex

    def find_nearest(
        self,
        vector: List[float],
        filter_expression: Optional[str] = None,
        limit: int = 10
    ) -> List[uuid.UUID]:
        """
        Executes a vector search against project templates combining vector similarity
        with an optional scalar query string filter (LanceDB SQL syntax style).
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
            results = query.limit(limit).to_pydantic(ProjectTemplateTableSchema)

            found_uuids = [uuid.UUID(row.id) for row in results]
            logger.info(f"Vector search on '{self._table_name}' returned {len(found_uuids)} matching records.")
            return found_uuids
        except Exception as ex:
            logger.error(f"Vector search failed on table '{self._table_name}': {ex}")
            raise VectorRepositoryException(f"Vector search failed on project templates: {str(ex)}") from ex