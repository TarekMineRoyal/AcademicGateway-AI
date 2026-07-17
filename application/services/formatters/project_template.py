from typing import List, Optional
from domain.models.project_template import ProjectTemplate


def format_project_document(
        template: ProjectTemplate,
        major_name: Optional[str],
        specialty_name: Optional[str],
        skill_names: List[str]
) -> str:
    """
    Compiles a project blueprint's structural scope, core requirements,
    and academic alignment targets into a clean narrative text block
    for vector database indexing.

    Args:
        template (ProjectTemplate): The project template domain read-model.
        major_name (Optional[str]): Text title of the target major, if aligned.
        specialty_name (Optional[str]): Text title of the target sub-track, if aligned.
        skill_names (List[str]): Text labels of required technical skills.

    Returns:
        str: A rich, un-prefixed narrative prose text block.
    """
    major_chunk = f"Targeted primarily toward {major_name} academic paths." if major_name else "Open to all general majors."
    specialty_chunk = f" Aligned with the {specialty_name} specialization track." if specialty_name else ""
    skills_chunk = ", ".join(skill_names) if skill_names else "General cross-functional capabilities"

    return (
        f"Academic Project Assignment Blueprint. "
        f"Project Title: {template.title}. "
        f"Core Description, Scope, and Deliverable Objectives: {template.description}. "
        f"Academic Requirements: {major_chunk}{specialty_chunk} "
        f"Required Student Technical Competencies: [{skills_chunk}]."
    )


def format_professor_advisor_query(
        title: str,
        description: str,
        major_name: Optional[str],
        specialty_name: Optional[str],
        skill_names: List[str]
) -> str:
    """
    Translates a project's core details into a targeted semantic search query
    to match against faculty professor profiles to find the ideal supervisor.
    """
    major_text = f" in the {major_name} department" if major_name else ""
    specialty_text = f" focusing on {specialty_name}" if specialty_name else ""
    skills_chunk = ", ".join(skill_names) if skill_names else "academic research methods"

    return (
        f"Identify an academic faculty professor and research advisor{major_text}{specialty_text} "
        f"whose expert research interests, industry backgrounds, and current lab direction "
        f"directly align with supervising a student project titled '{title}'. "
        f"The project entails: {description}. "
        f"The supervisor should have background relevant to these student competencies: [{skills_chunk}]."
    )