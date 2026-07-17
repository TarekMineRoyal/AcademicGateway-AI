from domain.models.research_interest import ResearchInterest


def format_research_interest_document(interest: ResearchInterest) -> str:
    """
    Transforms a global academic research interest area entity into a
    clean narrative description text block for vector database ingestion.

    Args:
        interest (ResearchInterest): The raw research interest entity from the domain layer.

    Returns:
        str: A clean, descriptive prose paragraph describing the research topic focus.
    """
    # Pure narrative output. The infrastructure layer handles adding any model prefixes later.
    return f"Academic faculty research interest area focus: {interest.area}"