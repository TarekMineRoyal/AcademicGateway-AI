from typing import List, Optional
from domain.models.student import Student


def format_student_document(
        student: Student,
        major_name: str,
        specialty_names: List[str],
        skill_names: List[str]
) -> str:
    """
    Compiles a student's full scholastic identity, degree tracks, technical
    competency matrix, and personal bio summary into a clean narrative text
    block for vector database indexing.

    Args:
        student (Student): The domain student read-model profile.
        major_name (str): The text title name of the student's primary major.
        specialty_names (List[str]): Text titles of any sub-track concentrations.
        skill_names (List[str]): Text labels of technical skills claimed by the student.

    Returns:
        str: A rich, un-prefixed narrative prose text block.
    """
    specialties_chunk = ", ".join(specialty_names) if specialty_names else "No specialized sub-tracks designated"
    skills_chunk = ", ".join(skill_names) if skill_names else "No technical skills indexed yet"
    about_me_chunk = student.about_me.strip() if student.about_me else "No professional biography or summary provided."

    return (
        f"Student Profile: {student.full_name}. "
        f"Primary Academic Major Path: {major_name}. "
        f"Sub-Track Concentrations: {specialties_chunk}. "
        f"Technical Competency Skill Matrix: {skills_chunk}. "
        f"Personal Background and Career Aspirations: {about_me_chunk}"
    )


def format_project_recommendation_query(
        major_name: str,
        specialty_names: List[str],
        skill_names: List[str],
        about_me: Optional[str]
) -> str:
    """
    Translates a student's active scholastic background and personal profile
    into a targeted semantic search query to match against project assignment blueprints.
    """
    specialties_chunk = ", ".join(specialty_names) if specialty_names else "general curriculum studies"
    skills_chunk = ", ".join(skill_names) if skill_names else "foundational academic concepts"
    about_chunk = about_me.strip() if about_me else "practical applied learning experiences"

    return (
        f"Find academic project assignment blueprints and research opportunities that strongly align "
        f"with a student majoring in {major_name}, pursuing concentrations in [{specialties_chunk}], "
        f"who possesses technical expertise in [{skills_chunk}], and is deeply interested in: {about_chunk}"
    )


def format_skill_recommendation_query(
        major_name: str,
        specialty_names: List[str],
        skill_names: List[str],
        about_me: Optional[str]
) -> str:
    """
    Translates a student's profile context into a targeted query designed to
    discover next-step, adjacent, or highly relevant skills they should acquire.
    """
    specialties_chunk = f" specializing in {', '.join(specialty_names)}" if specialty_names else ""
    skills_chunk = ", ".join(skill_names) if skill_names else "beginner academic tracks"
    about_chunk = f" Career focus: {about_me.strip()}" if about_me else ""

    return (
        f"Identify complementary industry skills, technical frameworks, or professional competencies "
        f"that are highly relevant or a natural progression for a {major_name} student{specialties_chunk} "
        f"who currently knows [{skills_chunk}].{about_chunk}"
    )