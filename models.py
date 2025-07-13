# models.py
from sqlalchemy import Column, Integer, String, Text, DateTime, Float, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from database import Base

class Content(Base):
    """
    Represents a piece of educational content.
    """
    __tablename__ = "content"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True, nullable=False)
    topic = Column(String, index=True, nullable=False)
    grade = Column(String, index=True, nullable=False) # e.g., "5", "High School", "University"
    file_name = Column(String, nullable=True) # Original file name
    content_text = Column(Text, nullable=False) # The actual text content
    embedding_vector = Column(Text, nullable=True) # Store as JSON string, or handle separately for FAISS
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # One-to-many relationship with QuestionAnswerLog (optional, for tracing)
    # qa_logs = relationship("QuestionAnswerLog", back_populates="content")

    def __repr__(self):
        return f"<Content(id={self.id}, title='{self.title}', topic='{self.topic}', grade='{self.grade}')>"

class QuestionAnswerLog(Base):
    """
    Logs user questions, AI answers, and associated metadata.
    """
    __tablename__ = "question_answer_logs"

    id = Column(Integer, primary_key=True, index=True)
    user_question = Column(Text, nullable=False)
    ai_answer = Column(Text, nullable=False)
    retrieved_content_ids = Column(Text, nullable=True) # Store as JSON string of IDs
    persona_used = Column(String, nullable=True)
    asked_at = Column(DateTime(timezone=True), server_default=func.now())

    # One-to-one relationship with Feedback
    feedback = relationship("Feedback", back_populates="qa_log", uselist=False)

    def __repr__(self):
        return f"<QuestionAnswerLog(id={self.id}, question='{self.user_question[:50]}...')>"

class Feedback(Base):
    """
    Stores user feedback on AI answers.
    """
    __tablename__ = "feedback"

    id = Column(Integer, primary_key=True, index=True)
    qa_log_id = Column(Integer, ForeignKey("question_answer_logs.id"), unique=True, nullable=False)
    rating = Column(Integer, nullable=False) # e.g., 1 (bad) to 5 (excellent), or 0/1 for thumbs up/down
    comment = Column(Text, nullable=True)
    submitted_at = Column(DateTime(timezone=True), server_default=func.now())

    # Many-to-one relationship with QuestionAnswerLog
    qa_log = relationship("QuestionAnswerLog", back_populates="feedback")

    def __repr__(self):
        return f"<Feedback(id={self.id}, qa_log_id={self.qa_log_id}, rating={self.rating})>"
