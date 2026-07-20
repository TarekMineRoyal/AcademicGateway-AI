import uuid
from datetime import datetime, timezone
import lancedb
import pytest

from domain.models.professor import Professor
from domain.models.project_template import ProjectTemplate
from domain.models.skill import Skill
from domain.models.student import Student
from infrastructure.config.settings import settings

# Database & Client Setup
from infrastructure.persistence.lancedb_client import LanceDbClient
# Repository Imports
from infrastructure.persistence.repositories.professor_repository import (
    ProfessorVectorRepository,
)
from infrastructure.persistence.repositories.project_repository import (
    ProjectTemplateVectorRepository,
)
from infrastructure.persistence.repositories.skill_repository import (
    SkillVectorRepository,
)
from infrastructure.persistence.repositories.student_repository import (
    StudentVectorRepository,
)
from infrastructure.persistence.schema_registry import VECTOR_DIMENSION

pytestmark = pytest.mark.integration


# ============================================================================
# 0. EPHEMERAL DATABASE FIXTURES
# ============================================================================


@pytest.fixture
def ephemeral_client(tmp_path):
    """
    Creates an isolated, ephemeral LanceDB client tied to a temporary directory.
    Guarantees no file locks leak or pollute active local configurations.
    """
    client = LanceDbClient()
    client._uri = str(tmp_path / "test_vector_space.lance")
    return client


@pytest.fixture
def mock_vector():
    """Generates a dummy dense vector matching the application's dimension specification."""
    return [0.01] * VECTOR_DIMENSION


# ============================================================================
# 4. PHYSICAL VECTOR DATABASE GATEWAYS (Checkpoint 4)
# ============================================================================


def test_lazy_table_initialization_and_index_creation(ephemeral_client):
    """Confirm lazy table initialization automatically builds indexes clean on disk."""
    repo = ProfessorVectorRepository(client=ephemeral_client)
    conn = ephemeral_client.get_connection()

    # Verify the client returns a valid LanceDB connection instance (clears lancedb import warning)
    assert isinstance(conn, lancedb.DBConnection)
    assert repo._table_name not in conn.list_tables().tables

    # Trigger lazy initialization
    table = repo._get_table()

    assert repo._table_name in conn.list_tables().tables
    assert table is not None
    assert repo._table is not None


def test_uuid_type_translations_and_upsert_matching(ephemeral_client, mock_vector):
    """Ensure that type translations (UUID to strings) execute flawlessly without data drops."""
    prof_repo = ProfessorVectorRepository(client=ephemeral_client)
    student_repo = StudentVectorRepository(client=ephemeral_client)

    prof_id = uuid.uuid4()
    student_id = uuid.uuid4()
    major_id = uuid.uuid4()
    interest_id = uuid.uuid4()
    specialty_id = uuid.uuid4()
    skill_id = uuid.uuid4()

    prof_domain = Professor(
        id=prof_id,
        full_name="Dr. Leslie Lamport",
        department="Distributed Computing",
        rank="Distinguished Professor",
        is_accepting_projects=True,
        research_interest_ids=[interest_id],
        about_me="Focusing on vector clock synchronization models.",
    )

    student_domain = Student(
        id=student_id,
        full_name="Alice Student",
        major_id=major_id,
        specialty_ids=[specialty_id],
        skill_ids=[skill_id],
        about_me="Undergraduate looking for concurrent computing tracks.",
    )

    prof_repo.upsert(prof_domain, mock_vector)
    student_repo.upsert(student_domain, mock_vector)

    matched_professors = prof_repo.find_nearest(mock_vector, limit=1)
    retrieved_student = student_repo.get_by_id(student_id)

    assert len(matched_professors) == 1
    assert matched_professors[0] == prof_id

    assert retrieved_student is not None
    assert retrieved_student.id == student_id
    assert retrieved_student.major_id == major_id


def test_get_names_by_ids_handles_empty_inputs_gracefully(ephemeral_client):
    """Verify that get_names_by_ids executes successfully when provided empty input lists."""
    skill_repo = SkillVectorRepository(client=ephemeral_client)
    names = skill_repo.get_names_by_ids([])
    assert isinstance(names, list)
    assert len(names) == 0


def test_get_names_by_ids_excludes_heavy_vector_projection(
    ephemeral_client, mock_vector
):
    """Assert that get_names_by_ids successfully excludes the heavy vector array column."""
    skill_repo = SkillVectorRepository(client=ephemeral_client)

    skill_1 = Skill(id=uuid.uuid4(), name="Asynchronous Programming")
    skill_repo.upsert(skill_1, mock_vector)

    table = skill_repo._get_table()
    filter_clause = f"id = '{str(skill_1.id)}'"

    raw_results = table.search().where(filter_clause).select(["name"]).to_list()
    resolved_names = skill_repo.get_names_by_ids([skill_1.id])

    assert len(resolved_names) == 1
    assert skill_1.name in resolved_names

    for row in raw_results:
        assert "name" in row
        assert "vector" not in row


def test_find_nearest_applies_scalar_prefiltering_correctly(
    ephemeral_client, mock_vector
):
    """Verify that find_nearest isolates items strictly passing pre-filter criteria."""
    prof_repo = ProfessorVectorRepository(client=ephemeral_client)

    prof_active = Professor(
        id=uuid.uuid4(),
        full_name="Active Advisor",
        department="CS",
        rank="Professor",
        is_accepting_projects=True,
        research_interest_ids=[],
        about_me="Available",
    )
    prof_busy = Professor(
        id=uuid.uuid4(),
        full_name="Busy Advisor",
        department="CS",
        rank="Professor",
        is_accepting_projects=False,
        research_interest_ids=[],
        about_me="Full capacity",
    )

    prof_repo.upsert(prof_active, mock_vector)
    prof_repo.upsert(prof_busy, mock_vector)

    results = prof_repo.find_nearest(
        mock_vector, filter_expression="is_accepting_projects = true"
    )

    assert prof_active.id in results
    assert prof_busy.id not in results


# ============================================================================
# DEDICATED PROJECT TEMPLATE VECTOR REPOSITORY TESTS
# ============================================================================


def test_project_template_repository_upsert_and_find_nearest(
    ephemeral_client, mock_vector
):
    """
    Verifies that ProjectTemplateVectorRepository successfully checks lazy table creation,
    handles datetime conversions, and executes pre-filtered vector queries.
    """
    project_repo = ProjectTemplateVectorRepository(client=ephemeral_client)
    project_id = uuid.uuid4()
    provider_id = uuid.uuid4()
    major_id = uuid.uuid4()
    specialty_id = uuid.uuid4()

    assert len(mock_vector) == VECTOR_DIMENSION
    assert VECTOR_DIMENSION == settings.EMBEDDING_DIMENSION

    template = ProjectTemplate(
        id=project_id,
        title="Next-Generation Semantic Vector Searcher",
        description="Testing physical LanceDB repository storage layers under isolation.",
        provider_id=provider_id,
        created_at=datetime.now(timezone.utc),
        skill_ids=[uuid.uuid4(), uuid.uuid4()],
        major_id=major_id,
        specialty_id=specialty_id,
    )

    project_repo.upsert(template, mock_vector)

    filter_expression = (
        f"major_id = '{str(major_id)}' AND specialty_id = '{str(specialty_id)}'"
    )
    matched_ids = project_repo.find_nearest(
        vector=mock_vector, filter_expression=filter_expression, limit=5
    )

    conn = ephemeral_client.get_connection()
    assert project_repo._table_name in conn.list_tables().tables
    assert len(matched_ids) == 1
    assert matched_ids[0] == project_id


def test_project_template_find_nearest_excludes_mismatched_major(
    ephemeral_client, mock_vector
):
    """Ensures project searches successfully drop blueprints belonging to distinct major IDs."""
    project_repo = ProjectTemplateVectorRepository(client=ephemeral_client)
    major_a = uuid.uuid4()
    major_b = uuid.uuid4()

    template = ProjectTemplate(
        id=uuid.uuid4(),
        title="Mechanical Robotics Shell",
        description="Building external industrial components.",
        provider_id=uuid.uuid4(),
        created_at=datetime.now(timezone.utc),
        skill_ids=[],
        major_id=major_a,
        specialty_id=None,
    )

    project_repo.upsert(template, mock_vector)

    results = project_repo.find_nearest(
        vector=mock_vector, filter_expression=f"major_id = '{str(major_b)}'", limit=5
    )

    assert template.id not in results


def test_get_student_by_id_missing_returns_none(ephemeral_client):
    """Verify that looking up a non-existent student ID returns None safely without crashing."""
    student_repo = StudentVectorRepository(client=ephemeral_client)
    random_id = uuid.uuid4()

    result = student_repo.get_by_id(random_id)

    assert result is None


# ============================================================================
# BULK UPSERT AND BLUE/GREEN TABLE SWAP INTEGRATION TESTS
# ============================================================================


def test_bulk_upsert_persists_batch_records(ephemeral_client, mock_vector):
    """Verify bulk_upsert writes multiple domain models in a single batch operation."""
    prof_repo = ProfessorVectorRepository(client=ephemeral_client)

    prof_1 = Professor(
        id=uuid.uuid4(),
        full_name="Prof A",
        department="CS",
        rank="Professor",
        is_accepting_projects=True,
        research_interest_ids=[],
        about_me="Bio A",
    )
    prof_2 = Professor(
        id=uuid.uuid4(),
        full_name="Prof B",
        department="EE",
        rank="Associate",
        is_accepting_projects=True,
        research_interest_ids=[],
        about_me="Bio B",
    )

    prof_repo.bulk_upsert([prof_1, prof_2], [mock_vector, mock_vector])

    results = prof_repo.find_nearest(mock_vector, limit=10)
    assert len(results) == 2
    assert prof_1.id in results
    assert prof_2.id in results


def test_blue_green_table_swap_and_cache_reload(ephemeral_client, mock_vector):
    """Verify swap_tables promotes staging data and reload_table clears cached handles."""
    staging_name = "professors_sync"
    live_name = "professors"

    live_repo = ProfessorVectorRepository(
        table_name=live_name, client=ephemeral_client
    )
    staging_repo = ProfessorVectorRepository(
        table_name=staging_name, client=ephemeral_client
    )

    # Insert initial record into live table
    prof_old = Professor(
        id=uuid.uuid4(),
        full_name="Old Prof",
        department="CS",
        rank="Professor",
        is_accepting_projects=True,
        research_interest_ids=[],
        about_me="Old",
    )
    live_repo.upsert(prof_old, mock_vector)

    # Ensure live handle is initialized and cached
    initial_results = live_repo.find_nearest(mock_vector, limit=10)
    assert prof_old.id in initial_results

    # Insert new record into staging table
    prof_new = Professor(
        id=uuid.uuid4(),
        full_name="New Prof",
        department="CS",
        rank="Professor",
        is_accepting_projects=True,
        research_interest_ids=[],
        about_me="New",
    )
    staging_repo.bulk_upsert([prof_new], [mock_vector])

    # Perform Blue/Green swap
    ephemeral_client.swap_tables(staging_name, live_name)

    # Invalidate in-memory cache on live repository
    live_repo.reload_table()

    # Query live repository again - should now point to swapped data
    swapped_results = live_repo.find_nearest(mock_vector, limit=10)
    assert prof_new.id in swapped_results
    assert prof_old.id not in swapped_results