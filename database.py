# database.py
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# SQLite database URL
SQLALCHEMY_DATABASE_URL = "sqlite:///./edurag.db"

# Create the SQLAlchemy engine
# connect_args={"check_same_thread": False} is needed for SQLite with FastAPI
# as SQLite operates on a single thread by default, and FastAPI uses multiple threads.
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)

# Create a SessionLocal class
# Each instance of SessionLocal will be a database session.
# The expire_on_commit=False prevents objects from expiring after commit,
# which can be useful when you want to access attributes after committing.
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for our declarative models
Base = declarative_base()

# Dependency to get a database session
# This function will be used in FastAPI path operations to get a DB session.
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
