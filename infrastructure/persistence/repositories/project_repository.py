import threading
import uuid
from typing import List, Optional

from lancedb.index import BTree

from application.interfaces.vector_repositories import IProjectTemplateVectorRepository
from domain.models.project_template import ProjectTemplate
from infrastructure.persistence.lancedb_client import lancedb_client
from infrastructure.persistence.schema_registry import ProjectTemplateTableSchema


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
            self._table = None

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

            table = conn.create_table(self._table_name, schema=ProjectTemplateTableSchema)
            table.create_index("id", config=BTree())
            table.create_index("major_id", config=BTree())
            table.create_index("specialty_id", config=BTree())

            self._table = table
            return self._table

    def upsert(self, template: ProjectTemplate, vector: List[float]) -> None:
        """
        Saves or updates a project template blueprint with its unified descriptive embedding.
        Uses LanceDB's high-speed merge_insert to ensure atomic upsert operations.
        """
        table = self._get_table()

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

    def bulk_upsert(
        self, templates: List[ProjectTemplate], vectors: List[List[float]]
    ) -> None:
        """
        Bulk upserts multiple project templates with their corresponding vectors in a single batch.
        """
        if not templates:
            return

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

    def delete(self, entity_id: uuid.UUID) -> None:
        """
        Removes a project blueprint vector record from persistent storage by its unique ID.
        Execution is natively idempotent in LanceDB.
        """
        table = self._get_table()
        table.delete(f"id = '{str(entity_id)}'")

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
        table = self._get_table()

        # Initialize the base vector query builder sequence
        query = table.search(vector)

        # Apply SQL-style predicate pre-filtering BEFORE setting the slice limit
        if filter_expression:
            query = query.where(filter_expression, prefilter=True)

        # Apply the final limit and execute the query builder pipeline
        results = query.limit(limit).to_pydantic(ProjectTemplateTableSchema)

        # Map string identifiers back into pure tracking domain UUIDs
        return [uuid.UUID(row.id) for row in results]