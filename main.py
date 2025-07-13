# main.py
import os
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form, status
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from typing import List, Optional
import uvicorn
import io

import models, schemas, crud, database, rag_pipeline, text_to_sql_pipeline

# Create the FastAPI app instance
app = FastAPI(
    title="EduRAG: Intelligent Tutor API",
    description="Backend API for an intelligent tutor using RAG and LangChain.",
    version="1.0.0",
)


# Create database tables on startup
@app.on_event("startup")
def on_startup():
    print("Creating database tables...")
    models.Base.metadata.create_all(bind=database.engine)
    print("Database tables created.")

    # Initialize the FAISS vector store with existing content
    db_session = next(database.get_db())
    rag_pipeline.initialize_vector_store(db_session)
    db_session.close()

    # Initialize the SQLDatabase for LangChain's text-to-SQL
    text_to_sql_pipeline.initialize_sql_database()


# Mount static files (for the interactive playground HTML)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def read_root():
    """
    Serves the interactive playground HTML page.
    """
    with open("static/index.html", "r") as f:
        return f.read()


@app.post(
    "/upload-content",
    response_model=schemas.ContentResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_content(
    file: UploadFile = File(
        ..., description="Text file (.txt) containing educational content."
    ),
    title: str = Form(
        ..., description="Title of the content (e.g., 'Introduction to Algebra')."
    ),
    topic: str = Form(
        ..., description="Topic of the content (e.g., 'Mathematics', 'Biology')."
    ),
    grade: str = Form(
        ...,
        description="Grade level for the content (e.g., '5th Grade', 'High School').",
    ),
    db: Session = Depends(database.get_db),
):
    """
    Uploads new textbook content with associated metadata.
    The content will be stored in the database and used to update the knowledge base for RAG.
    """
    if not file.filename.endswith(".txt"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only .txt files are allowed.",
        )

    try:
        content_bytes = await file.read()
        content_text = content_bytes.decode("utf-8")
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Could not read file content: {e}",
        )

    content_data = schemas.ContentCreate(title=title, topic=topic, grade=grade)
    db_content = crud.create_content(
        db, content=content_data, file_name=file.filename, content_text=content_text
    )

    # Add the new content to the FAISS vector store
    rag_pipeline.add_content_to_vector_store(db, db_content)

    return db_content


@app.post("/ask", response_model=schemas.AnswerResponse)
async def ask_question(
    request: schemas.QuestionRequest, db: Session = Depends(database.get_db)
):
    """
    Accepts a user question and returns an AI answer.
    If `is_sql_query` is true, it attempts to convert the natural language question into SQL.
    Otherwise, it uses the RAG pipeline to answer based on uploaded content.
    """
    ai_answer = ""
    retrieved_content_ids = []
    qa_log_id = -1  # Placeholder, will be updated after logging

    if request.is_sql_query:
        # Handle natural language SQL querying
        print(f"Received SQL query request: {request.question}")
        ai_answer = await text_to_sql_pipeline.query_database_natural_language(
            request.question
        )
        print(f"SQL query answer: {ai_answer}")
    else:
        # Handle RAG-based question answering
        print(
            f"Received RAG question request: {request.question} with persona: {request.persona}"
        )
        ai_answer, retrieved_content_ids = rag_pipeline.generate_rag_answer(
            request.question, request.persona
        )
        print(f"RAG answer: {ai_answer}")

    # Log the question and answer
    qa_log = crud.create_qa_log(
        db,
        user_question=request.question,
        ai_answer=ai_answer,
        retrieved_content_ids=retrieved_content_ids,
        persona_used=request.persona,
    )
    qa_log_id = qa_log.id

    return schemas.AnswerResponse(
        answer=ai_answer,
        qa_log_id=qa_log_id,
        retrieved_content_ids=retrieved_content_ids,
    )


@app.post(
    "/feedback",
    response_model=schemas.FeedbackResponse,
    status_code=status.HTTP_201_CREATED,
)
async def submit_feedback(
    feedback: schemas.FeedbackCreate, db: Session = Depends(database.get_db)
):
    """
    Allows users to rate the quality of AI responses.
    """
    qa_log = crud.get_qa_log(db, feedback.qa_log_id)
    if not qa_log:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Question-answer log not found.",
        )

    db_feedback = crud.create_feedback(db, feedback)
    return db_feedback


@app.get("/topics", response_model=List[schemas.TopicResponse])
async def get_topics(
    grade: Optional[str] = None,
    topic: Optional[str] = None,
    db: Session = Depends(database.get_db),
):
    """
    Filters and retrieves unique topics and their associated grades from the knowledge base.
    """
    topics_data = crud.get_topics_and_grades(db)
    filtered_topics = []
    for t in topics_data:
        if (grade is None or t.grade.lower() == grade.lower()) and (
            topic is None or t.topic.lower() == topic.lower()
        ):
            filtered_topics.append(t)
    return filtered_topics


@app.get("/metrics", response_model=schemas.MetricsResponse)
async def get_metrics(db: Session = Depends(database.get_db)):
    """
    Returns system statistics: total content files, unique topics, questions asked,
    feedback received, and average feedback rating.
    """
    return crud.get_metrics(db)
