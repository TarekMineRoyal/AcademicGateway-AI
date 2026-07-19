import uuid
from typing import List, Optional
import threading
import lancedb
from lancedb.index import BTree

from domain.models.student import Student
from application.interfaces.vector_repositories import IStudentVectorRepository
from infrastructure.persistence.lancedb_client import lancedb_client
from infrastructure.persistence.schema_registry import StudentTableSchema


class StudentVectorRepository(IStudentVectorRepository):
    """
    Concrete implementation of IStudentVectorRepository using LanceDB.
    Handles the mapping, persistence, and lookup of vectorized student read-models.
    """

    _repo_lock = threading.Lock()

    def __init__(self, client=lancedb_client) -> None:
        self._client = client
        self._table_name = "students"
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

            table = conn.create_table(self._table_name, schema=StudentTableSchema)
            table.create_index("id", config=BTree())

            self._table = table
            return self._table

    def upsert(self, student: Student, vector: List[float]) -> None:
        """
        Saves or updates a student profile record with its contextual embedding.
        Uses LanceDB's high-speed merge_insert to ensure atomic upsert operations.
        """
        table = self._get_table()

        # Translate pure domain model coordinates into the physical layout schema
        db_record = StudentTableSchema(
            id=str(student.id),
            full_name=student.full_name,
            major_id=str(student.major_id),
            specialty_ids=[str(sid) for sid in student.specialty_ids],
            skill_ids=[str(sid) for sid in student.skill_ids],
            about_me=student.about_me,
            vector=vector # type: ignore
        )

        # Execute thread-safe upsert matching the stringified primary identifier.
        # .model_dump() guarantees safe serialization compatibility into PyArrow layers.
        table.merge_insert(on="id") \
            .when_matched_update_all() \
            .when_not_matched_insert_all() \
            .execute([db_record.model_dump()])

    def get_by_id(self, student_id: uuid.UUID) -> Optional[Student]:
        """
        Fetches a flattened student profile to extract metadata tags for query setups.
        """
        table = self._get_table()

        # Thanks to the scalar index, this lookup evaluates instantly
        results = table.search() \
            .where(f"id = '{str(student_id)}'") \
            .limit(1) \
            .to_pydantic(StudentTableSchema)

        if not results:
            return None

        db_record = results[0]

        # Reconstruct and return the pure structural domain model object
        return Student(
            id=uuid.UUID(db_record.id),
            full_name=db_record.full_name,
            major_id=uuid.UUID(db_record.major_id),
            specialty_ids=[uuid.UUID(sid) for sid in db_record.specialty_ids],
            skill_ids=[uuid.UUID(sid) for sid in db_record.skill_ids],
            about_me=db_record.about_me
        )