from typing import List, Optional
from domain.models.project_template import ProjectTemplate


def _normalize_description(desc: str) -> str:
    """Helper to ensure the description is stripped and ends with exactly one period."""
    cleaned = desc.strip()
    return cleaned if cleaned.endswith(".") else f"{cleaned}."


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
    """
    major_chunk = f"Targeted primarily toward {major_name} academic paths." if major_name else "Open to all general majors."
    specialty_chunk = f" Aligned with the {specialty_name} specialization track." if specialty_name else ""
    skills_chunk = ", ".join(skill_names) if skill_names else "General cross-functional capabilities"

    # Ensure description doesn't cause double punctuation
    clean_desc = _normalize_description(template.description)

    return (
        f"Academic Project Assignment Blueprint. "
        f"Project Title: {template.title}. "
        f"Core Description, Scope, and Deliverable Objectives: {clean_desc} "
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

    # Ensure description doesn't cause double punctuation
    clean_desc = _normalize_description(description)

    return (
        f"Identify an academic faculty professor and research advisor{major_text}{specialty_text} "
        f"whose expert research interests, industry backgrounds, and current lab direction "
        f"directly align with supervising a student project titled '{title}'. "
        f"The project entails: {clean_desc} "
        f"The supervisor should have background relevant to these student competencies: [{skills_chunk}]."
    )