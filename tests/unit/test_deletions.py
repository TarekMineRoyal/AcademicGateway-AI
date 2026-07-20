import uuid
from unittest.mock import MagicMock

from application.commands.delete_professor import (
    DeleteProfessorCommand,
    DeleteProfessorCommandHandler,
)
from application.commands.delete_project import (
    DeleteProjectCommand,
    DeleteProjectCommandHandler,
)
from application.commands.delete_skill import (
    DeleteSkillCommand,
    DeleteSkillHandler,
)
from application.commands.delete_student import (
    DeleteStudentCommand,
    DeleteStudentCommandHandler,
)


def test_delete_student_command_handler_invokes_repository():
    mock_repo = MagicMock()
    handler = DeleteStudentCommandHandler(student_repository=mock_repo)
    target_id = uuid.uuid4()

    handler.handle(DeleteStudentCommand(id=target_id))

    mock_repo.delete.assert_called_once_with(target_id)


def test_delete_professor_command_handler_invokes_repository():
    mock_repo = MagicMock()
    handler = DeleteProfessorCommandHandler(professor_repository=mock_repo)
    target_id = uuid.uuid4()

    handler.handle(DeleteProfessorCommand(id=target_id))

    mock_repo.delete.assert_called_once_with(target_id)


def test_delete_project_command_handler_invokes_repository():
    mock_repo = MagicMock()
    handler = DeleteProjectCommandHandler(project_repository=mock_repo)
    target_id = uuid.uuid4()

    handler.handle(DeleteProjectCommand(id=target_id))

    mock_repo.delete.assert_called_once_with(target_id)


def test_delete_skill_command_handler_invokes_repository():
    mock_repo = MagicMock()
    handler = DeleteSkillHandler(skill_repository=mock_repo)
    target_id = uuid.uuid4()

    handler.handle(DeleteSkillCommand(id=target_id))

    mock_repo.delete.assert_called_once_with(target_id)