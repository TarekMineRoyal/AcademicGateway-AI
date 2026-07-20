from fastapi import Depends

# ---- Application Interface Contracts ----
from application.interfaces.embedding_service import IEmbeddingService
from application.interfaces.vector_repositories import (
    IProfessorVectorRepository,
    IProjectTemplateVectorRepository,
    ISkillVectorRepository,
    IStudentVectorRepository,
)

# ---- Application CQRS Single-Sync Command Handlers ----
from application.commands.sync_professor import SyncProfessorCommandHandler
from application.commands.sync_project import SyncProjectCommandHandler
from application.commands.sync_skill import SyncSkillHandler
from application.commands.sync_student import SyncStudentCommandHandler

# ---- Application CQRS Bulk-Sync Command Handlers ----
from application.commands.bulk_sync_professor import BulkSyncProfessorCommandHandler
from application.commands.bulk_sync_project import BulkSyncProjectCommandHandler
from application.commands.bulk_sync_skill import BulkSyncSkillCommandHandler
from application.commands.bulk_sync_student import BulkSyncStudentCommandHandler

# ---- Application CQRS Deletion Command Handlers ----
from application.commands.delete_professor import DeleteProfessorCommandHandler
from application.commands.delete_project import DeleteProjectCommandHandler
from application.commands.delete_skill import DeleteSkillHandler
from application.commands.delete_student import DeleteStudentCommandHandler

# ---- Application CQRS Query Handlers (Read-Path) ----
from application.queries.get_professor_suggestions import (
    GetProfessorSuggestionsQueryHandler,
)
from application.queries.get_project_recommendations import (
    GetProjectRecommendationsQueryHandler,
)
from application.queries.get_skill_recommendations import (
    GetSkillRecommendationsQueryHandler,
)

# ---- Infrastructure Implementations ----
from infrastructure.embedding.nomic_service import NomicEmbeddingService
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

# ==============================================================================
# GLOBAL SINGLETONS (Preserves Hot In-Memory Table Handle Caches)
# ==============================================================================
_embedding_service = NomicEmbeddingService()
_student_repository = StudentVectorRepository()
_project_repository = ProjectTemplateVectorRepository()
_professor_repository = ProfessorVectorRepository()
_skill_repository = SkillVectorRepository()


# ==============================================================================
# CORE INFRASTRUCTURE PROVIDERS
# ==============================================================================
def get_embedding_service() -> IEmbeddingService:
    """Provides the localized Nomic/SentenceTransformers embedding runtime engine."""
    return _embedding_service


def get_student_repository() -> IStudentVectorRepository:
    """Provides the persistent cached LanceDB student vector collection."""
    return _student_repository


def get_project_repository() -> IProjectTemplateVectorRepository:
    """Provides the persistent cached LanceDB project blueprint collection."""
    return _project_repository


def get_professor_repository() -> IProfessorVectorRepository:
    """Provides the persistent cached LanceDB faculty advisor collection."""
    return _professor_repository


def get_skill_repository() -> ISkillVectorRepository:
    """Provides the persistent cached LanceDB technical capabilities index."""
    return _skill_repository


# ==============================================================================
# CQRS SINGLE-SYNC COMMAND HANDLER PROVIDERS (Write-Path Mutations)
# ==============================================================================
def get_sync_professor_handler(
    embedder: IEmbeddingService = Depends(get_embedding_service),
    repository: IProfessorVectorRepository = Depends(get_professor_repository),
) -> SyncProfessorCommandHandler:
    return SyncProfessorCommandHandler(
        embedding_service=embedder, professor_repository=repository
    )


def get_sync_project_handler(
    embedder: IEmbeddingService = Depends(get_embedding_service),
    repository: IProjectTemplateVectorRepository = Depends(get_project_repository),
) -> SyncProjectCommandHandler:
    return SyncProjectCommandHandler(
        embedding_service=embedder, project_repository=repository
    )


def get_sync_skill_handler(
    embedder: IEmbeddingService = Depends(get_embedding_service),
    repository: ISkillVectorRepository = Depends(get_skill_repository),
) -> SyncSkillHandler:
    return SyncSkillHandler(embedding_service=embedder, skill_repository=repository)


def get_sync_student_handler(
    embedder: IEmbeddingService = Depends(get_embedding_service),
    repository: IStudentVectorRepository = Depends(get_student_repository),
) -> SyncStudentCommandHandler:
    return SyncStudentCommandHandler(
        embedding_service=embedder, student_repository=repository
    )


# ==============================================================================
# CQRS BULK-SYNC COMMAND HANDLER PROVIDERS (Backfill / Bulk Ingestion)
# ==============================================================================
def get_bulk_sync_professor_handler(
    embedder: IEmbeddingService = Depends(get_embedding_service),
    repository: IProfessorVectorRepository = Depends(get_professor_repository),
) -> BulkSyncProfessorCommandHandler:
    return BulkSyncProfessorCommandHandler(
        embedding_service=embedder, professor_repository=repository
    )


def get_bulk_sync_project_handler(
    embedder: IEmbeddingService = Depends(get_embedding_service),
    repository: IProjectTemplateVectorRepository = Depends(get_project_repository),
) -> BulkSyncProjectCommandHandler:
    return BulkSyncProjectCommandHandler(
        embedding_service=embedder, project_repository=repository
    )


def get_bulk_sync_skill_handler(
    embedder: IEmbeddingService = Depends(get_embedding_service),
    repository: ISkillVectorRepository = Depends(get_skill_repository),
) -> BulkSyncSkillCommandHandler:
    return BulkSyncSkillCommandHandler(
        embedding_service=embedder, skill_repository=repository
    )


def get_bulk_sync_student_handler(
    embedder: IEmbeddingService = Depends(get_embedding_service),
    repository: IStudentVectorRepository = Depends(get_student_repository),
) -> BulkSyncStudentCommandHandler:
    return BulkSyncStudentCommandHandler(
        embedding_service=embedder, student_repository=repository
    )


# ==============================================================================
# CQRS DELETION COMMAND HANDLER PROVIDERS (Write-Path Deletions)
# ==============================================================================
def get_delete_professor_handler(
    repository: IProfessorVectorRepository = Depends(get_professor_repository),
) -> DeleteProfessorCommandHandler:
    return DeleteProfessorCommandHandler(professor_repository=repository)


def get_delete_project_handler(
    repository: IProjectTemplateVectorRepository = Depends(get_project_repository),
) -> DeleteProjectCommandHandler:
    return DeleteProjectCommandHandler(project_repository=repository)


def get_delete_skill_handler(
    repository: ISkillVectorRepository = Depends(get_skill_repository),
) -> DeleteSkillHandler:
    return DeleteSkillHandler(skill_repository=repository)


def get_delete_student_handler(
    repository: IStudentVectorRepository = Depends(get_student_repository),
) -> DeleteStudentCommandHandler:
    return DeleteStudentCommandHandler(student_repository=repository)


# ==============================================================================
# CQRS QUERY HANDLER PROVIDERS (Read-Path Semantic Matches)
# ==============================================================================
def get_professor_suggestions_handler(
    embedder: IEmbeddingService = Depends(get_embedding_service),
    repository: IProfessorVectorRepository = Depends(get_professor_repository),
) -> GetProfessorSuggestionsQueryHandler:
    return GetProfessorSuggestionsQueryHandler(
        embedding_service=embedder, professor_repository=repository
    )


def get_project_recommendations_handler(
    embedder: IEmbeddingService = Depends(get_embedding_service),
    repository: IProjectTemplateVectorRepository = Depends(get_project_repository),
) -> GetProjectRecommendationsQueryHandler:
    return GetProjectRecommendationsQueryHandler(
        embedding_service=embedder, project_repository=repository
    )


def get_skill_recommendations_handler(
    embedder: IEmbeddingService = Depends(get_embedding_service),
    repository: ISkillVectorRepository = Depends(get_skill_repository),
) -> GetSkillRecommendationsQueryHandler:
    return GetSkillRecommendationsQueryHandler(
        embedding_service=embedder, skill_repository=repository
    )