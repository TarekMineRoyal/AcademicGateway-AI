import uuid
from typing import List
from fastapi import APIRouter, Depends, status

# ---- Dependency Graph Providers ----
from api.dependencies import (
    get_project_recommendations_handler,
    get_professor_suggestions_handler,
    get_skill_recommendations_handler,
)

# ---- CQRS Query Schema Handlers & DTOs ----
from application.queries.get_project_recommendations import (
    GetProjectRecommendationsQuery,
    GetProjectRecommendationsQueryHandler,
)
from application.queries.get_professor_suggestions import (
    GetProfessorSuggestionsQuery,
    GetProfessorSuggestionsQueryHandler,
)
from application.queries.get_skill_recommendations import (
    GetSkillRecommendationsQuery,
    GetSkillRecommendationsQueryHandler,
)

router = APIRouter()


@router.post(
    "/projects",
    status_code=status.HTTP_200_OK,
    response_model=List[uuid.UUID],
    summary="Fetch semantic project recommendations for a student."
)
def get_project_recommendations(
    query: GetProjectRecommendationsQuery,
    handler: GetProjectRecommendationsQueryHandler = Depends(get_project_recommendations_handler)
):
    """
    Accepts a student's profile context details and executes a semantic nearest-neighbor vector scan
    to fetch highly matching project blueprints[cite: 7].
    """
    return handler.handle(query)


@router.post(
    "/professors",
    status_code=status.HTTP_200_OK,
    response_model=List[uuid.UUID],
    summary="Discover closest matching faculty advisors for a project template."
)
def get_professor_suggestions(
    query: GetProfessorSuggestionsQuery,
    handler: GetProfessorSuggestionsQueryHandler = Depends(get_professor_suggestions_handler)
):
    """
    Translates a project template's requirements into an advisor-targeted semantic lookup,
    pre-filtering exclusively for active faculty members[cite: 6].
    """
    return handler.handle(query)


@router.post(
    "/skills",
    status_code=status.HTTP_200_OK,
    response_model=List[uuid.UUID],
    summary="Identify adjacent technical skills for student profile progression."
)
def get_skill_recommendations(
    query: GetSkillRecommendationsQuery,
    handler: GetSkillRecommendationsQueryHandler = Depends(get_skill_recommendations_handler)
):
    """
    Analyzes a student's current capability matrix and major track to isolate missing or
    highly relevant next-step technical skill nodes across the global catalog[cite: 8].
    """
    return handler.handle(query)