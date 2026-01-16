"""
Database initialization script for video analyzer conversation history.
This script creates a separate database and tables for conversation history.
Run this script to set up the PostgreSQL database.
"""
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database configuration
DB_USER = "pguser"
DB_PASSWORD = "NewSecurePassword2024"
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "video_analyzer_db"

def create_database():
    """Create the video_analyzer_db database if it doesn't exist."""
    try:
        # Connect to the default 'postgres' database
        conn = psycopg2.connect(
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT,
            database="postgres"
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        
        # Check if database exists
        cursor.execute(
            "SELECT 1 FROM pg_catalog.pg_database WHERE datname = %s",
            (DB_NAME,)
        )
        exists = cursor.fetchone()
        
        if not exists:
            cursor.execute(f"CREATE DATABASE {DB_NAME}")
            logger.info(f"✓ Created database: {DB_NAME}")
        else:
            logger.info(f"✓ Database {DB_NAME} already exists")
        
        cursor.close()
        conn.close()
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to create database: {e}")
        return False

def init_tables():
    """Initialize tables and indexes in the video_analyzer_db database."""
    try:
        # Connect to the new database
        conn = psycopg2.connect(
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT,
            database=DB_NAME
        )
        conn.autocommit = True
        cursor = conn.cursor()
        
        logger.info(f"Connected to database: {DB_NAME}")
        
        # Create conversation_history table
        create_table_query = """
        CREATE TABLE IF NOT EXISTS conversation_history (
            id SERIAL PRIMARY KEY,
            session_id VARCHAR(255) NOT NULL,
            role VARCHAR(50) NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
            content TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        cursor.execute(create_table_query)
        logger.info("✓ Created conversation_history table")
        
        # Create indexes
        create_index_session = """
        CREATE INDEX IF NOT EXISTS idx_conversation_session_id 
        ON conversation_history(session_id);
        """
        cursor.execute(create_index_session)
        logger.info("✓ Created index on session_id")
        
        create_index_timestamp = """
        CREATE INDEX IF NOT EXISTS idx_conversation_timestamp 
        ON conversation_history(timestamp);
        """
        cursor.execute(create_index_timestamp)
        logger.info("✓ Created index on timestamp")
        
        create_index_created_at = """
        CREATE INDEX IF NOT EXISTS idx_conversation_created_at 
        ON conversation_history(created_at);
        """
        cursor.execute(create_index_created_at)
        logger.info("✓ Created index on created_at")
        
        # Create cleanup function
        create_cleanup_function = """
        CREATE OR REPLACE FUNCTION cleanup_old_conversations()
        RETURNS INTEGER AS $$
        DECLARE
            deleted_count INTEGER;
        BEGIN
            DELETE FROM conversation_history 
            WHERE created_at < NOW() - INTERVAL '7 days';
            GET DIAGNOSTICS deleted_count = ROW_COUNT;
            RETURN deleted_count;
        END;
        $$ LANGUAGE plpgsql;
        """
        cursor.execute(create_cleanup_function)
        logger.info("✓ Created cleanup function for old conversations")
        
        # Create function to get message count
        create_count_function = """
        CREATE OR REPLACE FUNCTION get_session_message_count(p_session_id VARCHAR)
        RETURNS INTEGER AS $$
        DECLARE
            msg_count INTEGER;
        BEGIN
            SELECT COUNT(*) INTO msg_count
            FROM conversation_history
            WHERE session_id = p_session_id;
            RETURN msg_count;
        END;
        $$ LANGUAGE plpgsql;
        """
        cursor.execute(create_count_function)
        logger.info("✓ Created session message count function")
        
        cursor.close()
        conn.close()
        
        logger.info("\n✅ Database initialization completed successfully!")
        logger.info(f"Database: {DB_NAME}")
        logger.info("You can now start the backend server.")
        return True
        
    except Exception as e:
        logger.error(f"❌ Table initialization failed: {e}")
        return False

def main():
    """Main function to set up the database."""
    logger.info("=" * 60)
    logger.info("Video Analyzer - Database Initialization")
    logger.info("=" * 60)
    logger.info(f"Host: {DB_HOST}:{DB_PORT}")
    logger.info(f"User: {DB_USER}")
    logger.info(f"Database: {DB_NAME}")
    logger.info("=" * 60)
    
    # Step 1: Create database
    logger.info("\n[Step 1/2] Creating database...")
    if not create_database():
        logger.error("\n❌ Failed to create database. Please check:")
        logger.error("  1. PostgreSQL is running")
        logger.error("  2. User credentials are correct")
        logger.error("  3. PostgreSQL is accepting connections")
        return
    
    # Step 2: Initialize tables
    logger.info("\n[Step 2/2] Creating tables and indexes...")
    if not init_tables():
        logger.error("\n❌ Failed to initialize tables")
        return
    
    logger.info("\n" + "=" * 60)
    logger.info("✅ Setup Complete!")
    logger.info("=" * 60)
    logger.info("\nYou can now:")
    logger.info("  1. Start the backend: python main.py")
    logger.info("  2. Or run with uvicorn: uvicorn main:app --reload")

if __name__ == "__main__":
    main()
