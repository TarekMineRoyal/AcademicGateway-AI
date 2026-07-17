from domain.models.skill import Skill


def format_skill_document(skill: Skill) -> str:
    """
    Transforms a standardized technical or professional skill entity into a
    clean narrative description text block for vector database ingestion.

    Args:
        skill (Skill): The raw skill entity from the domain layer.

    Returns:
        str: A clean, descriptive prose paragraph describing the competency.
    """
    # We output pure, high-signal narrative text.
    # The infrastructure layer will handle adding model-specific token prefixes.
    return f"Technical and professional competency skill: {skill.name}"