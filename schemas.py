# schemas.py
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime

# --- Content Schemas ---
class ContentBase(BaseModel):
    title: str = Field(..., example="Introduction to Algebra")
    topic: str = Field(..., example="Mathematics")
    grade: str = Field(..., example="8th Grade")

class ContentCreate(ContentBase):
    # For file upload, content_text will be extracted from the file
    pass

class ContentResponse(ContentBase):
    id: int
    file_name: Optional[str] = None
    content_text: str
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True # For SQLAlchemy ORM compatibility

# --- Question Answering Schemas ---
class QuestionRequest(BaseModel):
    question: str = Field(..., example="What is photosynthesis?")
    persona: Optional[str] = Field("friendly", example="strict", description="Tutor persona (e.g., friendly, strict, humorous)")
    # Add a flag to indicate if it's a natural language SQL query
    is_sql_query: bool = Field(False, description="Set to true if the question is a natural language query about the database.")

class AnswerResponse(BaseModel):
    answer: str
    qa_log_id: int # ID of the logged question-answer pair for feedback
    retrieved_content_ids: Optional[List[int]] = None

# --- Feedback Schemas ---
class FeedbackCreate(BaseModel):
    qa_log_id: int = Field(..., description="The ID of the question-answer log entry being rated.")
    rating: int = Field(..., ge=1, le=5, description="Rating from 1 (poor) to 5 (excellent).")
    comment: Optional[str] = Field(None, example="The answer was very clear and helpful.")

class FeedbackResponse(FeedbackCreate):
    id: int
    submitted_at: datetime

    class Config:
        from_attributes = True

# --- Topic Filtering Schemas ---
class TopicFilter(BaseModel):
    grade: Optional[str] = None
    topic: Optional[str] = None

class TopicResponse(BaseModel):
    topic: str
    grade: str

    class Config:
        from_attributes = True

# --- Metrics Schemas ---
class MetricsResponse(BaseModel):
    total_content_files: int
    total_topics: int
    total_questions_asked: int
    total_feedback_received: int
    average_feedback_rating: Optional[float] = None
