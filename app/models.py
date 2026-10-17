from typing import List, Optional

from pydantic import BaseModel, Field


class StudentQuery(BaseModel):
    question: str = Field(..., min_length=1)
    student_level: Optional[str] = None
    subject: Optional[str] = None


class CourseRecommendation(BaseModel):
    course_name: str
    reason: str
    difficulty: Optional[str] = None
    topics: List[str] = Field(default_factory=list)


class TutorResponse(BaseModel):
    answer: str
    sources: List[str] = Field(default_factory=list)
    recommendations: List[CourseRecommendation] = Field(default_factory=list)