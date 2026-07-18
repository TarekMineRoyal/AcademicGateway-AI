from fastapi import APIRouter, Depends, status

# ---- Dependency Graph Providers ----
from api.dependencies import (
    get_sync_student_handler,
    get_sync_project_handler,
    get_sync_professor_handler,
    get_sync_skill_handler,
)

# ---- CQRS Command Schema Handlers & DTOs ----
from application.commands.sync_student import SyncStudentCommand, SyncStudentCommandHandler
from application.commands.sync_project import SyncProjectCommand, SyncProjectCommandHandler
from application.commands.sync_professor import SyncProfessorCommand, SyncProfessorCommandHandler
from application.commands.sync_skill import SyncSkillCommand, SyncSkillHandler

router = APIRouter()


@router.post(
    "/student",
    status_code=status.HTTP_200_OK,
    summary="Synchronizes or mutates a student profile vector space node."
)
async def sync_student(
    command: SyncStudentCommand,
    handler: SyncStudentCommandHandler = Depends(get_sync_student_handler)
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
    summary="Synchronizes or mutates a project blueprint vector space node."
)
async def sync_project(
    command: SyncProjectCommand,
    handler: SyncProjectCommandHandler = Depends(get_sync_project_handler)
):
    """
    Accepts an enriched project blueprint payload. Handles bulk ingestion, 
    real-time creation, and structural approval updates interchangeably.
    """
    handler.handle(command)
    return {"message": f"Project template {command.template.id} successfully synchronized."}


@router.post(
    "/professor",
    status_code=status.HTTP_200_OK,
    summary="Synchronizes or mutates a professor faculty vector space node."
)
async def sync_professor(
    command: SyncProfessorCommand,
    handler: SyncProfessorCommandHandler = Depends(get_sync_professor_handler)
):
    """
    Accepts an enriched professor details payload. Handles bulk sync ingestion 
    and direct lab/profile modifications interchangeably.
    """
    handler.handle(command)
    return {"message": f"Professor {command.professor.id} successfully synchronized."}


@router.post(
    "/skill",
    status_code=status.HTTP_200_OK,
    summary="Synchronizes or mutates a global competency vector space node."
)
async def sync_skill(
    command: SyncSkillCommand,
    handler: SyncSkillHandler = Depends(get_sync_skill_handler)
):
    """
    Accepts a raw technical skill capability payload. Handles system-wide catalog 
    population and real-time capability adjustments interchangeably.
    """
    handler.handle(command)
    return {"message": f"Skill {command.skill.id} successfully synchronized."}