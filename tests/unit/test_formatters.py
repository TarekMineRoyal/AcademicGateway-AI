import pytest
import uuid
from datetime import datetime, timezone

# Domain Model Imports
from domain.models.professor import Professor
from domain.models.project_template import ProjectTemplate
from domain.models.skill import Skill
from domain.models.student import Student

# Application Formatter Imports
from application.services.formatters.skill import format_skill_document
from application.services.formatters.student import (format_student_document, format_skill_recommendation_query, format_project_recommendation_query)
from application.services.formatters.professor import format_professor_document
from application.services.formatters.project_template import (format_project_document, format_professor_advisor_query)

pytestmark = pytest.mark.unit


# ============================================================================
# 1. PROFESSOR DOCUMENT FORMATTER TESTS
# ============================================================================

def test_format_professor_document_fully_populated():
    """Verify narrative layout matching when professor details are completely present."""
    prof = Professor(
        id=uuid.uuid4(),
        full_name="Dr. Alan Turing",
        department="Computer Science",
        rank="Full Professor",
        is_accepting_projects=True,
        research_interest_ids=[uuid.uuid4()],
        about_me="   Leading the Artificial Intelligence lab group.   "  # Intentional whitespace
    )
    interests = ["Machine Learning", "Quantum Computing"]

    result = format_professor_document(prof, interests)

    expected = (
        "Academic Faculty Professor Profile: Dr. Alan Turing. "
        "Department Division: Computer Science. "
        "Institutional Rank: Full Professor. "
        "Active Research Interest Domains: [Machine Learning, Quantum Computing]. "
        "Biography, Research Focus, and Lab Direction: Leading the Artificial Intelligence lab group."
    )
    assert result == expected


def test_format_professor_document_fallback_conditions():
    """Verify fallback strings trigger correctly when about_me is missing or empty arrays are passed."""
    prof = Professor(
        id=uuid.uuid4(),
        full_name="Dr. Empty Bio",
        department="Mathematics",
        rank="Associate Professor",
        is_accepting_projects=True,
        about_me=None
    )

    result = format_professor_document(prof, interest_areas=[])

    expected = (
        "Academic Faculty Professor Profile: Dr. Empty Bio. "
        "Department Division: Mathematics. "
        "Institutional Rank: Associate Professor. "
        "Active Research Interest Domains: [General academic supervision and instruction]. "
        "Biography, Research Focus, and Lab Direction: No professional biography or lab summary provided."
    )
    assert result == expected


# ============================================================================
# 2. PROJECT DOCUMENT & ADVISOR QUERY TESTS
# ============================================================================

def test_format_project_document_fully_populated():
    """Verify project document narrative generation with all structural details present."""
    template = ProjectTemplate(
        id=uuid.uuid4(),
        title="Autonomous Drone Routing",
        description="Developing multi-agent pathfinding systems.",
        provider_id=uuid.uuid4(),
        created_at=datetime.now(timezone.utc)
    )

    result = format_project_document(
        template=template,
        major_name="Robotics Engineering",
        specialty_name="Autonomous Systems",
        skill_names=["Python", "ROS", "C++"]
    )

    expected = (
        "Academic Project Assignment Blueprint. "
        "Project Title: Autonomous Drone Routing. "
        "Core Description, Scope, and Deliverable Objectives: Developing multi-agent pathfinding systems. "
        "Academic Requirements: Targeted primarily toward Robotics Engineering academic paths. Aligned with the Autonomous Systems specialization track. "
        "Required Student Technical Competencies: [Python, ROS, C++]."
    )
    assert result == expected


def test_format_project_document_fallbacks():
    """Ensure optional alignments leave the output structurally valid without ghost spacing."""
    template = ProjectTemplate(
        id=uuid.uuid4(),
        title="Open Optimization Task",
        description="General algorithm tweaking.",
        provider_id=uuid.uuid4(),
        created_at=datetime.now(timezone.utc)
    )

    result = format_project_document(
        template=template,
        major_name=None,
        specialty_name=None,
        skill_names=[]
    )

    expected = (
        "Academic Project Assignment Blueprint. "
        "Project Title: Open Optimization Task. "
        "Core Description, Scope, and Deliverable Objectives: General algorithm tweaking. "
        "Academic Requirements: Open to all general majors. "
        "Required Student Technical Competencies: [General cross-functional capabilities]."
    )
    assert result == expected


def test_format_professor_advisor_query_fully_populated():
    """Verify search query building when targeting specific departments and expertise profiles."""
    result = format_professor_advisor_query(
        title="Neural Style Transfer",
        description="Applying deep learning models to artwork translation.",
        major_name="Data Science",
        specialty_name="Computer Vision",
        skill_names=["PyTorch", "OpenCV"]
    )

    expected = (
        "Identify an academic faculty professor and research advisor in the Data Science department focusing on Computer Vision "
        "whose expert research interests, industry backgrounds, and current lab direction "
        "directly align with supervising a student project titled 'Neural Style Transfer'. "
        "The project entails: Applying deep learning models to artwork translation. "
        "The supervisor should have background relevant to these student competencies: [PyTorch, OpenCV]."
    )
    assert result == expected


def test_format_professor_advisor_query_fallbacks():
    """Ensure generalized queries drop specific context seamlessly without breaking spacing contracts."""
    result = format_professor_advisor_query(
        title="Basic Coding Lab",
        description="Introductory programming exercises.",
        major_name=None,
        specialty_name=None,
        skill_names=[]
    )

    expected = (
        "Identify an academic faculty professor and research advisor "
        "whose expert research interests, industry backgrounds, and current lab direction "
        "directly align with supervising a student project titled 'Basic Coding Lab'. "
        "The project entails: Introductory programming exercises. "
        "The supervisor should have background relevant to these student competencies: [academic research methods]."
    )
    assert result == expected


# ============================================================================
# 3. SKILL DOCUMENT FORMATTER TESTS
# ============================================================================

def test_format_skill_document():
    """Verify that pure structural skill text layout compiles exactly as specified."""
    skill = Skill(id=uuid.uuid4(), name="Kubernetes Orchestration")
    result = format_skill_document(skill)
    assert result == "Technical and professional competency skill: Kubernetes Orchestration"


# ============================================================================
# 4. STUDENT DOCUMENT & RECOMMENDATION QUERY TESTS
# ============================================================================

def test_format_student_document_fully_populated():
    """Verify full student profile compiles cleanly into an un-prefixed narrative block."""
    student = Student(
        id=uuid.uuid4(),
        full_name="Grace Hopper",
        major_id=uuid.uuid4(),
        about_me="   Enthusiastic compiler architecture designer.   "
    )

    result = format_student_document(
        student=student,
        major_name="Software Engineering",
        specialty_names=["Systems Software", "Compilers"],
        skill_names=["Assembly", "C", "Linkers"]
    )

    expected = (
        "Student Profile: Grace Hopper. "
        "Primary Academic Major Path: Software Engineering. "
        "Sub-Track Concentrations: Systems Software, Compilers. "
        "Technical Competency Skill Matrix: Assembly, C, Linkers. "
        "Personal Background and Career Aspirations: Enthusiastic compiler architecture designer."
    )
    assert result == expected


def test_format_student_document_fallbacks():
    """Verify fallbacks render successfully when a student profile lacks specialized text metadata."""
    student = Student(
        id=uuid.uuid4(),
        full_name="Freshman Student",
        major_id=uuid.uuid4(),
        about_me=None
    )

    result = format_student_document(
        student=student,
        major_name="Undecided Engineering",
        specialty_names=[],
        skill_names=[]
    )

    expected = (
        "Student Profile: Freshman Student. "
        "Primary Academic Major Path: Undecided Engineering. "
        "Sub-Track Concentrations: No specialized sub-tracks designated. "
        "Technical Competency Skill Matrix: No technical skills indexed yet. "
        "Personal Background and Career Aspirations: No professional biography or summary provided."
    )
    assert result == expected


def test_format_project_recommendation_query_fully_populated():
    """Verify the student semantic matcher query structure when targeting project templates."""
    result = format_project_recommendation_query(
        major_name="Bioinformatics",
        specialty_names=["Genomics", "Data Analytics"],
        skill_names=["R Programming", "BLAST Search"],
        about_me="   Looking to map DNA sequencing pathways.   "
    )

    expected = (
        "Find academic project assignment blueprints and research opportunities that strongly align "
        "with a student majoring in Bioinformatics, pursuing concentrations in [Genomics, Data Analytics], "
        "who possesses technical expertise in [R Programming, BLAST Search], and is deeply interested in: Looking to map DNA sequencing pathways."
    )
    assert result == expected


def test_format_project_recommendation_query_fallbacks():
    """Ensure clean narrative layout when the matching criteria fallback values are invoked."""
    result = format_project_recommendation_query(
        major_name="General Studies",
        specialty_names=[],
        skill_names=[],
        about_me=None
    )

    expected = (
        "Find academic project assignment blueprints and research opportunities that strongly align "
        "with a student majoring in General Studies, pursuing concentrations in [general curriculum studies], "
        "who possesses technical expertise in [foundational academic concepts], and is deeply interested in: practical applied learning experiences"
    )
    assert result == expected


def test_format_skill_recommendation_query_fully_populated():
    """Verify adjacent-skill discovering search query syntax matches exactly."""
    result = format_skill_recommendation_query(
        major_name="Cybersecurity",
        specialty_names=["Penetration Testing", "Network Security"],
        skill_names=["Wireshark", "Linux"],
        about_me="   Aspiring white-hat defense auditor.   "
    )

    expected = (
        "Identify complementary industry skills, technical frameworks, or professional competencies "
        "that are highly relevant or a natural progression for a Cybersecurity student specializing in Penetration Testing, Network Security "
        "who currently knows [Wireshark, Linux]. Career focus: Aspiring white-hat defense auditor."
    )
    assert result == expected


def test_format_skill_recommendation_query_fallbacks():
    """Confirm text normalization behaves appropriately for skills matching without biographical extensions."""
    result = format_skill_recommendation_query(
        major_name="Business",
        specialty_names=[],
        skill_names=[],
        about_me=None
    )

    expected = (
        "Identify complementary industry skills, technical frameworks, or professional competencies "
        "that are highly relevant or a natural progression for a Business student "
        "who currently knows [beginner academic tracks]."
    )
    assert result == expected