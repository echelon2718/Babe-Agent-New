from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
import os
from sqlalchemy.exc import SQLAlchemyError
from dotenv import load_dotenv
from contextlib import contextmanager

load_dotenv()

DB_URL = f"mysql+pymysql://{os.getenv('DB_USERNAME')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_DATABASE')}"


engine = create_engine(DB_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

Base = declarative_base()


@contextmanager
def get_db_session():
    """Context manager untuk session yang aman"""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except SQLAlchemyError as e:
        session.rollback()
        print(f"[DB ERROR] {e}")
        raise
    except Exception as e:
        session.rollback()
        print(f"[GENERAL ERROR] {e}")
        raise
    finally:
        session.close()