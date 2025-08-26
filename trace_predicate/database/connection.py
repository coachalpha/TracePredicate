"""
Database connection and session management for TracePredicate.

This module provides database connection utilities, session management,
and initialization functions for the TracePredicate PostgreSQL database.
"""

import logging
import os
from contextlib import contextmanager
from typing import Generator, Optional

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from .models import Base

logger = logging.getLogger(__name__)


class DatabaseConfig:
    """Database configuration management."""
    
    def __init__(self):
        # Default database settings
        self.host = os.getenv("DB_HOST", "localhost")
        self.port = os.getenv("DB_PORT", "5432")
        self.database = os.getenv("DB_NAME", "tracepredicate")
        self.username = os.getenv("DB_USER", "postgres")
        self.password = os.getenv("DB_PASSWORD", "")
        
        # Connection pool settings
        self.pool_size = int(os.getenv("DB_POOL_SIZE", "10"))
        self.max_overflow = int(os.getenv("DB_MAX_OVERFLOW", "20"))
        self.pool_timeout = int(os.getenv("DB_POOL_TIMEOUT", "30"))
        self.pool_recycle = int(os.getenv("DB_POOL_RECYCLE", "3600"))
        
        # SSL settings
        self.ssl_mode = os.getenv("DB_SSL_MODE", "prefer")
    
    @property
    def database_url(self) -> str:
        """Get the database connection URL."""
        if self.password:
            auth = f"{self.username}:{self.password}"
        else:
            auth = self.username
            
        url = f"postgresql://{auth}@{self.host}:{self.port}/{self.database}"
        
        if self.ssl_mode != "prefer":
            url += f"?sslmode={self.ssl_mode}"
            
        return url
    
    @property
    def test_database_url(self) -> str:
        """Get the test database connection URL."""
        test_db_name = f"{self.database}_test"
        
        if self.password:
            auth = f"{self.username}:{self.password}"
        else:
            auth = self.username
            
        return f"postgresql://{auth}@{self.host}:{self.port}/{test_db_name}"


class DatabaseManager:
    """Database connection and session management."""
    
    def __init__(self, config: Optional[DatabaseConfig] = None):
        self.config = config or DatabaseConfig()
        self._engine: Optional[Engine] = None
        self._session_factory: Optional[sessionmaker] = None
    
    @property
    def engine(self) -> Engine:
        """Get or create the database engine."""
        if self._engine is None:
            self._engine = self._create_engine()
        return self._engine
    
    def _create_engine(self) -> Engine:
        """Create the SQLAlchemy engine with connection pooling."""
        logger.info(f"Creating database engine for {self.config.host}:{self.config.port}/{self.config.database}")
        
        engine = create_engine(
            self.config.database_url,
            pool_size=self.config.pool_size,
            max_overflow=self.config.max_overflow,
            pool_timeout=self.config.pool_timeout,
            pool_recycle=self.config.pool_recycle,
            pool_pre_ping=True,  # Validate connections before use
            echo=os.getenv("DB_ECHO", "false").lower() == "true"  # Log SQL queries if requested
        )
        
        return engine
    
    @property
    def session_factory(self) -> sessionmaker:
        """Get or create the session factory."""
        if self._session_factory is None:
            self._session_factory = sessionmaker(bind=self.engine)
        return self._session_factory
    
    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """
        Get a database session with automatic cleanup.
        
        Usage:
            with db_manager.get_session() as session:
                # Use session for database operations
                session.query(Device).all()
        """
        session = self.session_factory()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()
    
    def create_tables(self) -> None:
        """Create all database tables."""
        try:
            logger.info("Creating database tables...")
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database tables created successfully")
        except SQLAlchemyError as e:
            logger.error(f"Error creating database tables: {e}")
            raise
    
    def drop_tables(self) -> None:
        """Drop all database tables."""
        try:
            logger.warning("Dropping all database tables...")
            Base.metadata.drop_all(bind=self.engine)
            logger.info("Database tables dropped successfully")
        except SQLAlchemyError as e:
            logger.error(f"Error dropping database tables: {e}")
            raise
    
    def create_database(self) -> None:
        """Create the database if it doesn't exist."""
        # Connect to postgres database to create our target database
        postgres_url = self.config.database_url.replace(f"/{self.config.database}", "/postgres")
        
        try:
            temp_engine = create_engine(postgres_url)
            
            with temp_engine.connect() as conn:
                # Check if database exists
                result = conn.execute(
                    text("SELECT 1 FROM pg_database WHERE datname = :db_name"),
                    {"db_name": self.config.database}
                )
                
                if not result.fetchone():
                    # Database doesn't exist, create it
                    conn.execute(text("COMMIT"))  # End any open transaction
                    conn.execute(text(f'CREATE DATABASE "{self.config.database}"'))
                    logger.info(f"Created database: {self.config.database}")
                else:
                    logger.info(f"Database already exists: {self.config.database}")
            
            temp_engine.dispose()
            
        except SQLAlchemyError as e:
            logger.error(f"Error creating database: {e}")
            raise
    
    def test_connection(self) -> bool:
        """Test the database connection."""
        try:
            with self.get_session() as session:
                session.execute(text("SELECT 1"))
            logger.info("Database connection test successful")
            return True
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            return False
    
    def get_database_info(self) -> dict:
        """Get database information and statistics."""
        try:
            with self.get_session() as session:
                # Get PostgreSQL version
                version_result = session.execute(text("SELECT version()"))
                version = version_result.scalar()
                
                # Get database size
                size_result = session.execute(
                    text("SELECT pg_size_pretty(pg_database_size(:db_name))"),
                    {"db_name": self.config.database}
                )
                size = size_result.scalar()
                
                # Get table information
                tables_result = session.execute(text("""
                    SELECT 
                        schemaname,
                        tablename,
                        pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
                    FROM pg_tables 
                    WHERE schemaname = 'public'
                    ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC
                """))
                
                tables = [
                    {"schema": row[0], "name": row[1], "size": row[2]}
                    for row in tables_result.fetchall()
                ]
                
                return {
                    "version": version,
                    "database_size": size,
                    "tables": tables,
                    "connection_pool": {
                        "size": self.engine.pool.size(),
                        "checked_in": self.engine.pool.checkedin(),
                        "checked_out": self.engine.pool.checkedout()
                    }
                }
                
        except SQLAlchemyError as e:
            logger.error(f"Error getting database info: {e}")
            return {}
    
    def close(self) -> None:
        """Close all database connections."""
        if self._engine:
            self._engine.dispose()
            logger.info("Database connections closed")


# Global database manager instance
_db_manager: Optional[DatabaseManager] = None


def get_database_manager(config: Optional[DatabaseConfig] = None) -> DatabaseManager:
    """Get the global database manager instance."""
    global _db_manager
    
    if _db_manager is None:
        _db_manager = DatabaseManager(config)
    
    return _db_manager


@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """
    Convenience function to get a database session.
    
    Usage:
        from trace_predicate.database import get_db_session
        
        with get_db_session() as session:
            devices = session.query(Device).all()
    """
    db_manager = get_database_manager()
    with db_manager.get_session() as session:
        yield session


def init_database(drop_existing: bool = False) -> None:
    """
    Initialize the database with tables.
    
    Args:
        drop_existing: Whether to drop existing tables first
    """
    logger.info("Initializing TracePredicate database...")
    
    db_manager = get_database_manager()
    
    # Create database if it doesn't exist
    db_manager.create_database()
    
    # Test connection
    if not db_manager.test_connection():
        raise Exception("Failed to connect to database")
    
    # Drop tables if requested
    if drop_existing:
        db_manager.drop_tables()
    
    # Create tables
    db_manager.create_tables()
    
    logger.info("Database initialization completed successfully")


def main():
    """Test database connection and setup."""
    import sys
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Initialize database
        init_database()
        
        # Get database info
        db_manager = get_database_manager()
        info = db_manager.get_database_info()
        
        print("Database Information:")
        print(f"Version: {info.get('version', 'Unknown')}")
        print(f"Size: {info.get('database_size', 'Unknown')}")
        print(f"Tables: {len(info.get('tables', []))}")
        
        if info.get('tables'):
            print("\nTable Information:")
            for table in info['tables'][:10]:  # Show first 10 tables
                print(f"  {table['name']}: {table['size']}")
        
        print("\nDatabase setup completed successfully!")
        
    except Exception as e:
        logger.error(f"Database setup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()