from fastapi import APIRouter, BackgroundTasks, Depends, status

# ---- Dependency Graph Providers ----
from api.dependencies import (
    get_bulk_sync_professor_handler,
    get_bulk_sync_project_handler,
    get_bulk_sync_skill_handler,
    get_bulk_sync_student_handler,
    get_sync_professor_handler,
    get_sync_project_handler,
    get_sync_skill_handler,
    get_sync_student_handler,
)

# ---- CQRS Single-Sync Command Schema Handlers & DTOs ----
from application.commands.sync_professor import (
    SyncProfessorCommand,
    SyncProfessorCommandHandler,
)
from application.commands.sync_project import (
    SyncProjectCommand,
    SyncProjectCommandHandler,
)
from application.commands.sync_skill import SyncSkillCommand, SyncSkillHandler
from application.commands.sync_student import (
    SyncStudentCommand,
    SyncStudentCommandHandler,
)

# ---- CQRS Bulk-Sync Command Schema Handlers & DTOs ----
from application.commands.bulk_sync_professor import (
    BulkSyncProfessorCommand,
    BulkSyncProfessorCommandHandler,
)
from application.commands.bulk_sync_project import (
    BulkSyncProjectCommand,
    BulkSyncProjectCommandHandler,
)
from application.commands.bulk_sync_skill import (
    BulkSyncSkillCommand,
    BulkSyncSkillCommandHandler,
)
from application.commands.bulk_sync_student import (
    BulkSyncStudentCommand,
    BulkSyncStudentCommandHandler,
)

router = APIRouter()


# ==============================================================================
# SINGLE-EVENT SYNCHRONIZATION ENDPOINTS (Real-Time Live Updates)
# ==============================================================================


@router.post(
    "/student",
    status_code=status.HTTP_200_OK,
    summary="Synchronizes or mutates a student profile vector space node.",
)
async def sync_student(
    command: SyncStudentCommand,
    handler: SyncStudentCommandHandler = Depends(get_sync_student_handler),
):
    """
    Accepts an enriched student payload. Handles initial database synchronization,
    new registration entries, and profile edits interchangeably.
    """
    handler.handle(command)
    return {"message": f"Student {command.student.id} successfully synchronized."}


@router.post(
    "/project",
    status_code=status.HTTP_200_OK,
    summary="Synchronizes or mutates a project blueprint vector space node.",
)
async def sync_project(
    command: SyncProjectCommand,
    handler: SyncProjectCommandHandler = Depends(get_sync_project_handler),
):
    """
    Accepts an enriched project blueprint payload. Handles real-time creation
    and structural approval updates interchangeably.
    """
    handler.handle(command)
    return {
        "message": f"Project template {command.template.id} successfully synchronized."
    }


@router.post(
    "/professor",
    status_code=status.HTTP_200_OK,
    summary="Synchronizes or mutates a professor faculty vector space node.",
)
async def sync_professor(
    command: SyncProfessorCommand,
    handler: SyncProfessorCommandHandler = Depends(get_sync_professor_handler),
):
    """
    Accepts an enriched professor details payload. Handles direct lab/profile
    modifications.
    """
    handler.handle(command)
    return {"message": f"Professor {command.professor.id} successfully synchronized."}


@router.post(
    "/skill",
    status_code=status.HTTP_200_OK,
    summary="Synchronizes or mutates a global competency vector space node.",
)
async def sync_skill(
    command: SyncSkillCommand,
    handler: SyncSkillHandler = Depends(get_sync_skill_handler),
):
    """
    Accepts a raw technical skill capability payload. Handles system-wide catalog
    population and real-time capability adjustments interchangeably.
    """
    handler.handle(command)
    return {"message": f"Skill {command.skill.id} successfully synchronized."}


# ==============================================================================
# BULK SYNCHRONIZATION ENDPOINTS (Blue/Green Backfill Ingestion)
# ==============================================================================


@router.post(
    "/bulk/student",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Asynchronously queues bulk student profile synchronization pipeline.",
)
async def bulk_sync_student(
    command: BulkSyncStudentCommand,
    background_tasks: BackgroundTasks,
    handler: BulkSyncStudentCommandHandler = Depends(get_bulk_sync_student_handler),
):
    """
    Accepts a batch collection of student profile payloads. Offloads VRAM-safe chunk processing
    and Blue/Green table swapping to a background task, returning 202 Accepted immediately.
    """
    background_tasks.add_task(handler.handle, command)
    return {
        "status": "accepted",
        "message": f"Bulk student synchronization pipeline queued for {len(command.items)} records.",
    }


@router.post(
    "/bulk/project",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Asynchronously queues bulk project template synchronization pipeline.",
)
async def bulk_sync_project(
    command: BulkSyncProjectCommand,
    background_tasks: BackgroundTasks,
    handler: BulkSyncProjectCommandHandler = Depends(get_bulk_sync_project_handler),
):
    """
    Accepts a batch collection of project template payloads. Offloads VRAM-safe chunk processing
    and Blue/Green table swapping to a background task, returning 202 Accepted immediately.
    """
    background_tasks.add_task(handler.handle, command)
    return {
        "status": "accepted",
        "message": f"Bulk project template synchronization pipeline queued for {len(command.items)} records.",
    }


@router.post(
    "/bulk/professor",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Asynchronously queues bulk professor faculty synchronization pipeline.",
)
async def bulk_sync_professor(
    command: BulkSyncProfessorCommand,
    background_tasks: BackgroundTasks,
    handler: BulkSyncProfessorCommandHandler = Depends(
        get_bulk_sync_professor_handler
    ),
):
    """
    Accepts a batch collection of professor faculty payloads. Offloads VRAM-safe chunk processing
    and Blue/Green table swapping to a background task, returning 202 Accepted immediately.
    """
    background_tasks.add_task(handler.handle, command)
    return {
        "status": "accepted",
        "message": f"Bulk professor synchronization pipeline queued for {len(command.items)} records.",
    }


@router.post(
    "/bulk/skill",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Asynchronously queues bulk skill catalog synchronization pipeline.",
)
async def bulk_sync_skill(
    command: BulkSyncSkillCommand,
    background_tasks: BackgroundTasks,
    handler: BulkSyncSkillCommandHandler = Depends(get_bulk_sync_skill_handler),
):
    """
    Accepts a batch collection of skill catalog payloads. Offloads VRAM-safe chunk processing
    and Blue/Green table swapping to a background task, returning 202 Accepted immediately.
    """
    background_tasks.add_task(handler.handle, command)
    return {
        "status": "accepted",
        "message": f"Bulk skill synchronization pipeline queued for {len(command.items)} records.",
    }