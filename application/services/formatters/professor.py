from typing import List
from domain.models.professor import Professor


def format_professor_document(professor: Professor, interest_areas: List[str]) -> str:
    """
    Compiles a faculty member's institutional role, academic department,
    active research domains, and personal lab bio summary into a clean
    narrative text block for vector database indexing.

    Args:
        professor (Professor): The professor domain read-model.
        interest_areas (List[str]): Text labels of the professor's active research interests.

    Returns:
        str: A rich, un-prefixed narrative prose text block.
    """
    interests_chunk = ", ".join(interest_areas) if interest_areas else "General academic supervision and instruction"
    about_me_chunk = professor.about_me.strip() if professor.about_me else "No professional biography or lab summary provided."

    return (
        f"Academic Faculty Professor Profile: {professor.full_name}. "
        f"Department Division: {professor.department}. "
        f"Institutional Rank: {professor.rank}. "
        f"Active Research Interest Domains: [{interests_chunk}]. "
        f"Biography, Research Focus, and Lab Direction: {about_me_chunk}"
    )