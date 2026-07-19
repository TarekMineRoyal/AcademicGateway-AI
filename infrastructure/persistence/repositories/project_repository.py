import uuid
from typing import List, Optional
import threading

from domain.models.project_template import ProjectTemplate
from application.interfaces.vector_repositories import IProjectTemplateVectorRepository
from infrastructure.persistence.lancedb_client import lancedb_client
from infrastructure.persistence.schema_registry import ProjectTemplateTableSchema


class ProjectTemplateVectorRepository(IProjectTemplateVectorRepository):
    """
    Concrete implementation of IProjectTemplateVectorRepository using LanceDB.
    Handles the mapping, persistence, and semantic searches for project blueprints.
    """

    _repo_lock = threading.Lock()

    def __init__(self, client=lancedb_client) -> None:
        self._client = client
        self._table_name = "project_templates"
        self._table = None  # In-memory handle cache to avoid continuous disk I/O

    def _get_table(self):
        if self._table is not None:
            return self._table

        conn = self._client.get_connection()

        if self._table_name in conn.table_names():
            self._table = conn.open_table(self._table_name)
            return self._table

        with self._repo_lock:
            if self._table_name in conn.table_names():
                self._table = conn.open_table(self._table_name)
                return self._table

            table = conn.create_table(self._table_name, schema=ProjectTemplateTableSchema)
            table.create_scalar_index("id")
            table.create_scalar_index("major_id")
            table.create_scalar_index("specialty_id")

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