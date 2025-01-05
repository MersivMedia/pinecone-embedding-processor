import sqlite3
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/init.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def init_db():
    """Initialize SQLite database."""
    try:
        conn = sqlite3.connect('processed_files.db')
        c = conn.cursor()
        
        # Track overall file processing
        c.execute('''CREATE TABLE IF NOT EXISTS processed_files
                     (filename text PRIMARY KEY, 
                      processed_date text,
                      total_pages integer,
                      current_page integer,
                      chunks integer,
                      namespace text,
                      status text)''')
        
        # Track page-level progress
        c.execute('''CREATE TABLE IF NOT EXISTS processed_pages
                     (filename text,
                      page_number integer,
                      processed_date text,
                      chunks integer,
                      PRIMARY KEY (filename, page_number))''')
        
        conn.commit()
        logger.info("Database initialized successfully")
        
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")
        raise
    finally:
        conn.close()

if __name__ == "__main__":
    Path('logs').mkdir(exist_ok=True)
    init_db() 