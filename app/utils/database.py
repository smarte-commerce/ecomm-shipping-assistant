from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from typing import Generator
import os

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./shopping_assistant.db")

# Create engine with proper configuration
if DATABASE_URL.startswith("sqlite"):
    engine = create_engine(
        DATABASE_URL, 
        connect_args={"check_same_thread": False},
        echo=False  # Set to True for SQL query logging
    )
else:
    engine = create_engine(DATABASE_URL, echo=False)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()

def get_db() -> Generator[Session, None, None]:
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def create_tables():
    """Create all database tables"""
    from app.models.product import Product
    from app.models.user import User  
    from app.models.discount import Discount
    from app.models.conversation import Conversation, Message, Order
    
    Base.metadata.create_all(bind=engine)

def drop_tables():
    """Drop all database tables"""
    Base.metadata.drop_all(bind=engine)

class DatabaseManager:
    """Database manager for advanced operations"""
    
    @staticmethod
    def init_db():
        """Initialize database with default data"""
        create_tables()
        
    @staticmethod
    def reset_db():
        """Reset database (drop and recreate)"""
        drop_tables()
        create_tables()
        
    @staticmethod
    def get_session() -> Session:
        """Get a new database session"""
        return SessionLocal()
        
    @staticmethod
    def close_session(session: Session):
        """Close database session"""
        session.close() 