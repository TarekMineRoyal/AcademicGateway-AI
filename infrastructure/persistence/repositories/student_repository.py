import logging
import threading
import uuid
from typing import List, Optional

from lancedb.index import BTree

from application.exceptions.application_exceptions import VectorRepositoryException
from application.interfaces.vector_repositories import IStudentVectorRepository
from domain.models.student import Student
from infrastructure.persistence.lancedb_client import lancedb_client
from infrastructure.persistence.schema_registry import StudentTableSchema

logger = logging.getLogger(__name__)


class StudentVectorRepository(IStudentVectorRepository):
    """
    Concrete implementation of IStudentVectorRepository using LanceDB.
    Handles the mapping, persistence, lookup, and deletion of vectorized student read-models.
    """

    _repo_lock = threading.Lock()

    def __init__(self, table_name: Optional[str] = None, client=lancedb_client) -> None:
        self._client = client
        self._table_name = table_name or "students"
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
                table = conn.create_table(self._table_name, schema=StudentTableSchema)
                table.create_index("id", config=BTree())

                self._table = table
                logger.info(f"Table '{self._table_name}' initialized successfully with BTree indexes.")
                return self._table
        except Exception as ex:
            logger.error(f"Failed to access or create table '{self._table_name}': {ex}")
            raise VectorRepositoryException(f"Error initializing table '{self._table_name}': {str(ex)}") from ex

    def upsert(self, student: Student, vector: List[float]) -> None:
        """
        Saves or updates a student profile record with its contextual embedding.
        Uses LanceDB's high-speed merge_insert to ensure atomic upsert operations.
        """
        try:
            table = self._get_table()
            logger.info(f"Upserting student profile ID '{student.id}' ({student.full_name})")

            # Translate pure domain model coordinates into the physical layout schema
            db_record = StudentTableSchema(
                id=str(student.id),
                full_name=student.full_name,
                major_id=str(student.major_id),
                specialty_ids=[str(sid) for sid in student.specialty_ids],
                skill_ids=[str(sid) for sid in student.skill_ids],
                about_me=student.about_me,
                vector=vector  # type: ignore
            )

            # Execute thread-safe upsert matching the stringified primary identifier.
            # .model_dump() guarantees safe serialization compatibility into PyArrow layers.
            table.merge_insert(on="id") \
                .when_matched_update_all() \
                .when_not_matched_insert_all() \
                .execute([db_record.model_dump()])

            logger.info(f"Successfully upserted student ID '{student.id}'.")
        except Exception as ex:
            logger.error(f"Failed to upsert student ID '{student.id}': {ex}")
            raise VectorRepositoryException(f"Upsert failed for student '{student.id}': {str(ex)}") from ex

    def bulk_upsert(self, students: List[Student], vectors: List[List[float]]) -> None:
        """
        Bulk upserts multiple student records with their corresponding vectors in a single batch.
        """
        if not students:
            logger.warning("Bulk upsert invoked with an empty list of students. Skipping execution.")
            return

        batch_count = len(students)
        logger.info(f"Starting bulk upsert for {batch_count} student records.")

        try:
            table = self._get_table()

            records = [
                StudentTableSchema(
                    id=str(st.id),
                    full_name=st.full_name,
                    major_id=str(st.major_id),
                    specialty_ids=[str(sid) for sid in st.specialty_ids],
                    skill_ids=[str(sid) for sid in st.skill_ids],
                    about_me=st.about_me,
                    vector=vec,  # type: ignore
                ).model_dump()
                for st, vec in zip(students, vectors)
            ]

            table.merge_insert(on="id") \
                .when_matched_update_all() \
                .when_not_matched_insert_all() \
                .execute(records)

            logger.info(f"Successfully completed bulk upsert for {batch_count} students.")
        except Exception as ex:
            logger.error(f"Failed bulk upsert for batch of size {batch_count}: {ex}")
            raise VectorRepositoryException(f"Bulk upsert failed: {str(ex)}") from ex

    def delete(self, entity_id: uuid.UUID) -> None:
        """
        Removes a student profile vector record from persistent storage by its unique ID.
        Execution is natively idempotent in LanceDB.
        """
        try:
            table = self._get_table()
            str_id = str(entity_id)
            logger.info(f"Deleting student vector record with ID '{str_id}'")
            table.delete(f"id = '{str_id}'")
            logger.info(f"Deleted student record ID '{str_id}' successfully.")
        except Exception as ex:
            logger.error(f"Failed to delete student record ID '{entity_id}': {ex}")
            raise VectorRepositoryException(f"Deletion failed for student '{entity_id}': {str(ex)}") from ex

    def get_by_id(self, student_id: uuid.UUID) -> Optional[Student]:
        """
        Fetches a flattened student profile to extract metadata tags for query setups.
        """
        try:
            table = self._get_table()
            str_id = str(student_id)
            logger.info(f"Fetching student record by ID '{str_id}'")

            # Thanks to the scalar index, this lookup evaluates instantly
            results = table.search() \
                .where(f"id = '{str_id}'") \
                .limit(1) \
                .to_pydantic(StudentTableSchema)

            if not results:
                logger.info(f"No student record found for ID '{str_id}'.")
                return None

            db_record = results[0]
            logger.info(f"Successfully fetched student record for ID '{str_id}'.")

            # Reconstruct and return the pure structural domain model object
            return Student(
                id=uuid.UUID(db_record.id),
                full_name=db_record.full_name,
                major_id=uuid.UUID(db_record.major_id),
                specialty_ids=[uuid.UUID(sid) for sid in db_record.specialty_ids],
                skill_ids=[uuid.UUID(sid) for sid in db_record.skill_ids],
                about_me=db_record.about_me
            )
        except Exception as ex:
            logger.error(f"Failed to fetch student record by ID '{student_id}': {ex}")
            raise VectorRepositoryException(f"Fetch failed for student '{student_id}': {str(ex)}") from ex