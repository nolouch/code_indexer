from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from setting.base import DATABASE_URI, SESSION_POOL_SIZE
import logging

logger = logging.getLogger(__name__)

# Only create engine if DATABASE_URI is provided
if DATABASE_URI is not None:
    engine = create_engine(
        DATABASE_URI,
        pool_size=SESSION_POOL_SIZE,
        max_overflow=20,
        pool_timeout=30,
        pool_recycle=240,
        pool_pre_ping=True,
    )
    
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
else:
    logger.warning("DATABASE_URI is None, database functionality will be disabled")
    engine = None
    SessionLocal = None

Base = declarative_base()
