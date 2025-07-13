# crud.py
from sqlalchemy.orm import Session
from sqlalchemy import func, desc
from typing import List, Optional, Dict, Any
import json

import models, schemas

def create_content(db: Session, content: schemas.ContentCreate, file_name: Optional[str] = None, content_text: str = ""):
    """
    Creates a new content entry in the database.
    """
    db_content = models.Content(
        title=content.title,
        topic=content.topic,
        grade=content.grade,
        file_name=file_name,
        content_text=content_text
    )
    db.add(db_content)
    db.commit()
    db.refresh(db_content)
    return db_content

def get_content(db: Session, content_id: int):
    """
    Retrieves a content entry by its ID.
    """
    return db.query(models.Content).filter(models.Content.id == content_id).first()

def get_contents(db: Session, skip: int = 0, limit: int = 100):
    """
    Retrieves multiple content entries.
    """
    return db.query(models.Content).offset(skip).limit(limit).all()

def get_content_by_topic_and_grade(db: Session, topic: Optional[str] = None, grade: Optional[str] = None):
    """
    Retrieves content filtered by topic and/or grade.
    """
    query = db.query(models.Content)
    if topic:
        query = query.filter(models.Content.topic == topic)
    if grade:
        query = query = query.filter(models.Content.grade == grade)
    return query.all()

def create_qa_log(db: Session, user_question: str, ai_answer: str, retrieved_content_ids: Optional[List[int]] = None, persona_used: Optional[str] = None):
    """
    Logs a question-answer pair.
    """
    db_qa_log = models.QuestionAnswerLog(
        user_question=user_question,
        ai_answer=ai_answer,
        retrieved_content_ids=json.dumps(retrieved_content_ids) if retrieved_content_ids else None,
        persona_used=persona_used
    )
    db.add(db_qa_log)
    db.commit()
    db.refresh(db_qa_log)
    return db_qa_log

def get_qa_log(db: Session, qa_log_id: int):
    """
    Retrieves a question-answer log by its ID.
    """
    return db.query(models.QuestionAnswerLog).filter(models.QuestionAnswerLog.id == qa_log_id).first()

def create_feedback(db: Session, feedback: schemas.FeedbackCreate):
    """
    Creates feedback for a question-answer log.
    Ensures only one feedback per qa_log_id.
    """
    # Check if feedback already exists for this qa_log_id
    existing_feedback = db.query(models.Feedback).filter(models.Feedback.qa_log_id == feedback.qa_log_id).first()
    if existing_feedback:
        # Optionally update existing feedback instead of raising error
        existing_feedback.rating = feedback.rating
        existing_feedback.comment = feedback.comment
        existing_feedback.submitted_at = func.now()
        db.commit()
        db.refresh(existing_feedback)
        return existing_feedback
    else:
        db_feedback = models.Feedback(
            qa_log_id=feedback.qa_log_id,
            rating=feedback.rating,
            comment=feedback.comment
        )
        db.add(db_feedback)
        db.commit()
        db.refresh(db_feedback)
        return db_feedback

def get_feedback(db: Session, feedback_id: int):
    """
    Retrieves feedback by its ID.
    """
    return db.query(models.Feedback).filter(models.Feedback.id == feedback_id).first()

def get_topics_and_grades(db: Session) -> List[schemas.TopicResponse]:
    """
    Retrieves all unique topics and their associated grades from the content.
    """
    # Query distinct topic and grade pairs
    topics_data = db.query(models.Content.topic, models.Content.grade).distinct().all()
    return [schemas.TopicResponse(topic=t, grade=g) for t, g in topics_data]

def get_metrics(db: Session) -> schemas.MetricsResponse:
    """
    Calculates and returns various system metrics.
    """
    total_content_files = db.query(models.Content).count()
    total_topics = db.query(models.Content.topic).distinct().count()
    total_questions_asked = db.query(models.QuestionAnswerLog).count()
    total_feedback_received = db.query(models.Feedback).count()

    average_feedback_rating = None
    if total_feedback_received > 0:
        avg_rating = db.query(func.avg(models.Feedback.rating)).scalar()
        average_feedback_rating = round(avg_rating, 2)

    return schemas.MetricsResponse(
        total_content_files=total_content_files,
        total_topics=total_topics,
        total_questions_asked=total_questions_asked,
        total_feedback_received=total_feedback_received,
        average_feedback_rating=average_feedback_rating
    )
