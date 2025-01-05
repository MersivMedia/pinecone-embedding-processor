import os
import uuid
import logging
import time
from typing import List, Optional
from datetime import datetime
from pathlib import Path
import sqlite3
import shutil
import psutil
import io
import base64
import gc
import pikepdf
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from config.websites import WEBSITE_LIST, MAX_DEPTH, EXCLUDE_PATTERNS

from anthropic import Anthropic
from pinecone import Pinecone
from openai import OpenAI
from dotenv import load_dotenv
from pdf2image import convert_from_bytes

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/processor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize clients
try:
    anthropic = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    pc = Pinecone(
        api_key=os.getenv("PINECONE_API_KEY"),
        environment=os.getenv("PINECONE_ENVIRONMENT")
    )
    index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))
    logger.info("Successfully initialized all clients")
except Exception as e:
    logger.error(f"Error initializing clients: {str(e)}")
    raise

def extract_text_with_retry(image, max_retries=3):
    """Extract text from image with retry logic."""
    for attempt in range(max_retries):
        try:
            # Convert image to bytes
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            img_b64 = base64.b64encode(img_byte_arr).decode('utf-8')

            # Try extraction with different prompts
            prompts = [
                "Extract all text from this image, maintaining paragraphs and formatting.",
                "Please transcribe the text content from this image.",
                "What text do you see in this image? Please transcribe it exactly."
            ]

            response = anthropic.messages.create(
                model=os.getenv("ANTHROPIC_MODEL"),
                max_tokens=1500,
                temperature=0,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": img_b64
                            }
                        },
                        {
                            "type": "text",
                            "text": prompts[attempt]
                        }
                    ]
                }]
            )
            return response.content[0].text
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            logger.warning(f"Retry {attempt + 1} failed: {str(e)}")
            time.sleep(1)

def create_embedding(text: str) -> List[float]:
    """Create embedding using OpenAI's API."""
    try:
        response = openai_client.embeddings.create(
            input=text,
            model=os.getenv("OPENAI_EMBEDDING_MODEL")
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Error creating embedding: {str(e)}")
        raise

def get_text_chunks(text: str, chunk_size: int = 1000) -> List[str]:
    """Split text into chunks."""
    words = text.split()
    chunks = []
    current_chunk = []
    current_size = 0
    
    for word in words:
        current_size += len(word) + 1  # +1 for space
        if current_size > chunk_size:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_size = len(word)
        else:
            current_chunk.append(word)
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def init_db():
    """Initialize SQLite database."""
    conn = sqlite3.connect('processed_files.db')
    c = conn.cursor()
    try:
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
    finally:
        conn.close()

def is_already_processed(filename: str) -> bool:
    """Check if file has already been processed."""
    conn = sqlite3.connect('processed_files.db')
    c = conn.cursor()
    try:
        c.execute('SELECT status FROM processed_files WHERE filename = ? AND status = "completed"', (filename,))
        return c.fetchone() is not None
    finally:
        conn.close()

def mark_as_processed(filename: str, total_pages: int, chunks: int, namespace: str):
    """Mark file as processed in database."""
    conn = sqlite3.connect('processed_files.db')
    c = conn.cursor()
    try:
        c.execute('''INSERT OR REPLACE INTO processed_files 
                     (filename, processed_date, total_pages, current_page, chunks, namespace, status)
                     VALUES (?, datetime('now'), ?, ?, ?, ?, 'completed')''',
                  (filename, total_pages, total_pages, chunks, namespace))
        conn.commit()
        logger.info(f"Marked {filename} as completed in database")
    finally:
        conn.close()

def update_processing_status(filename: str, status: str, current_page: int, total_pages: int = None):
    """Update processing status."""
    conn = sqlite3.connect('processed_files.db')
    c = conn.cursor()
    try:
        if total_pages is not None:
            c.execute('''INSERT OR REPLACE INTO processed_files 
                         (filename, processed_date, total_pages, current_page, status)
                         VALUES (?, datetime('now'), ?, ?, ?)''',
                      (filename, total_pages, current_page, status))
        else:
            c.execute('''UPDATE processed_files 
                         SET current_page = ?, status = ?, processed_date = datetime('now')
                         WHERE filename = ?''',
                      (current_page, status, filename))
        conn.commit()
        logger.info(f"Updated status for {filename}: {status}, page {current_page}")
    finally:
        conn.close()

def mark_page_processed(filename: str, page_number: int, chunks: int):
    """Mark a page as processed."""
    conn = sqlite3.connect('processed_files.db')
    c = conn.cursor()
    try:
        c.execute('''INSERT OR REPLACE INTO processed_pages
                     (filename, page_number, processed_date, chunks)
                     VALUES (?, ?, datetime('now'), ?)''',
                  (filename, page_number, chunks))
        conn.commit()
        logger.info(f"Marked page {page_number} as processed for {filename}")
    finally:
        conn.close()

def process_pdf(pdf_bytes: bytes, filename: str, namespace: str) -> tuple[int, list]:
    """Process a single PDF file."""
    try:
        logger.info(f"Starting to process {filename}")
        
        # Check if file was partially processed
        is_complete, last_page, total_pages = get_processing_status(filename)
        if is_complete:
            logger.info(f"File {filename} was already completely processed")
            return 0, []
            
        # Open PDF using pikepdf
        pdf = pikepdf.Pdf.open(io.BytesIO(pdf_bytes))
        total_pages = len(pdf.pages)
        chunk_texts = []
        total_chunks = 0
        
        if last_page == 0:
            update_processing_status(filename, 'in_progress', last_page, total_pages)
        
        # Process one page at a time
        for page_num in range(last_page + 1, total_pages + 1):
            try:
                logger.info(f"Processing page {page_num}/{total_pages} of {filename}")
                
                # Convert single page with memory limits
                images = convert_from_bytes(
                    pdf_bytes,
                    first_page=page_num,
                    last_page=page_num,
                    thread_count=1  # Reduce thread count
                )
                
                if not images:
                    logger.error(f"Failed to convert page {page_num}")
                    continue
                    
                image = images[0]
                
                # Extract text and immediately clear image data
                text = extract_text_with_retry(image)
                image.close()
                del images
                del image
                gc.collect()
                
                if not text:
                    logger.warning(f"No text extracted from page {page_num}")
                    continue
                
                # Process chunks in smaller batches
                chunks = get_text_chunks(text, chunk_size=500)  # Reduced chunk size
                page_chunks = 0
                
                for chunk_num, chunk in enumerate(chunks, 1):
                    try:
                        embedding = create_embedding(chunk)
                        
                        index.upsert(
                            vectors=[{
                                'id': f"{namespace}-p{page_num}-c{chunk_num}",
                                'values': embedding,
                                'metadata': {
                                    'text': chunk,
                                    'page': page_num,
                                    'filename': filename
                                }
                            }]
                        )
                        
                        total_chunks += 1
                        page_chunks += 1
                        logger.info(f"Processed chunk {chunk_num} from page {page_num}")
                        
                        # Force garbage collection every few chunks
                        if chunk_num % 5 == 0:
                            gc.collect()
                        
                    except Exception as e:
                        logger.error(f"Error processing chunk {chunk_num} from page {page_num}: {str(e)}")
                
                # Mark progress and cleanup
                mark_page_processed(filename, page_num, page_chunks)
                update_processing_status(filename, 'in_progress', page_num)
                logger.info(f"Completed page {page_num} with {page_chunks} chunks")
                
                # Force cleanup after each page
                gc.collect()
                
                # Add small delay between pages to prevent memory buildup
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error processing page {page_num}: {str(e)}")
                continue
                
        pdf.close()
        del pdf
        gc.collect()
        
        logger.info(f"Completed processing {filename} with {total_chunks} total chunks")
        return total_chunks, chunk_texts
        
    except Exception as e:
        logger.error(f"Error processing {filename}: {str(e)}")
        return 0, []

def get_processing_status(filename: str) -> tuple[bool, int, int]:
    """Get processing status and last processed page."""
    conn = sqlite3.connect('processed_files.db')
    c = conn.cursor()
    try:
        c.execute('''SELECT status, current_page, total_pages 
                     FROM processed_files 
                     WHERE filename = ?''', (filename,))
        result = c.fetchone()
        
        if result is None:
            return False, 0, 0
        return result[0] == 'completed', result[1] or 0, result[2] or 0
    finally:
        conn.close()

def check_memory_usage():
    """Monitor memory usage and log if it's getting too high."""
    process = psutil.Process()
    memory_info = process.memory_info()
    memory_percent = process.memory_percent()
    
    # Get system-wide memory info
    system_memory = psutil.virtual_memory()
    
    logger.info(
        f"Process Memory: {memory_info.rss / 1024 / 1024:.2f}MB ({memory_percent:.1f}%) | "
        f"System Memory: Used {system_memory.used / 1024 / 1024:.2f}MB "
        f"({system_memory.percent:.1f}%)"
    )
    
    # List top memory processes if usage is high
    if memory_percent > 70:
        logger.warning("High memory usage detected! Top processes:")
        for proc in sorted(psutil.process_iter(['pid', 'name', 'memory_percent']), 
                          key=lambda x: x.info['memory_percent'] or 0, 
                          reverse=True)[:5]:
            logger.warning(f"{proc.info['name']}: {proc.info['memory_percent']:.1f}%")
    
    return memory_percent < 90

def is_pdf_url(url: str) -> bool:
    """Check if URL points to a PDF file."""
    try:
        response = requests.head(url, allow_redirects=True)
        content_type = response.headers.get('content-type', '').lower()
        return 'application/pdf' in content_type or url.lower().endswith('.pdf')
    except Exception as e:
        logger.error(f"Error checking URL {url}: {str(e)}")
        return False

def extract_text_from_html(html_content: str) -> str:
    """Extract meaningful text from HTML content."""
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Remove script and style elements
    for element in soup(['script', 'style', 'header', 'footer', 'nav']):
        element.decompose()
    
    # Get text
    text = soup.get_text(separator=' ', strip=True)
    
    # Clean up whitespace
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = ' '.join(chunk for chunk in chunks if chunk)
    
    return text

def get_valid_links(soup: BeautifulSoup, base_url: str) -> set:
    """Extract valid links from HTML content."""
    valid_links = set()
    
    for link in soup.find_all('a', href=True):
        url = urljoin(base_url, link['href'])
        
        # Skip if URL matches exclude patterns
        if any(pattern in url for pattern in EXCLUDE_PATTERNS):
            continue
            
        # Only include links from the same domain
        if urlparse(url).netloc == urlparse(base_url).netloc:
            valid_links.add(url)
            
    return valid_links

def process_website(url: str, depth: int = 0, processed_urls: set = None) -> tuple[int, list]:
    """Process a website and its linked pages."""
    if processed_urls is None:
        processed_urls = set()
        
    if depth > MAX_DEPTH or url in processed_urls:
        return 0, []
        
    processed_urls.add(url)
    total_chunks = 0
    chunk_texts = []
    
    try:
        logger.info(f"Processing URL: {url} (depth {depth})")
        
        if is_pdf_url(url):
            # Download and process PDF
            response = requests.get(url)
            pdf_bytes = response.content
            filename = url.split('/')[-1] or f"webpage_{uuid.uuid4()}.pdf"
            namespace = f"{datetime.now().strftime('%Y%m%d')}-{filename}"
            return process_pdf(pdf_bytes, filename, namespace)
            
        # Process HTML content
        response = requests.get(url)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        text = extract_text_from_html(response.text)
        
        # Process text chunks
        chunks = get_text_chunks(text)
        filename = f"webpage_{uuid.uuid4()}.html"
        namespace = f"{datetime.now().strftime('%Y%m%d')}-{filename}"
        
        for chunk_num, chunk in enumerate(chunks, 1):
            try:
                embedding = create_embedding(chunk)
                
                index.upsert(
                    vectors=[{
                        'id': f"{namespace}-c{chunk_num}",
                        'values': embedding,
                        'metadata': {
                            'text': chunk,
                            'url': url,
                            'type': 'webpage'
                        }
                    }]
                )
                
                total_chunks += 1
                chunk_texts.append(chunk)
                
            except Exception as e:
                logger.error(f"Error processing chunk {chunk_num} from {url}: {str(e)}")
        
        # Process linked pages if not at max depth
        if depth < MAX_DEPTH:
            valid_links = get_valid_links(soup, url)
            for link in valid_links:
                sub_chunks, sub_texts = process_website(link, depth + 1, processed_urls)
                total_chunks += sub_chunks
                chunk_texts.extend(sub_texts)
        
        return total_chunks, chunk_texts
        
    except Exception as e:
        logger.error(f"Error processing website {url}: {str(e)}")
        return 0, []

def watch_directory_and_websites(input_dir: str, processed_dir: str, error_dir: str):
    """Watch directory for PDFs and process websites from config."""
    init_db()
    
    while True:
        try:
            # Process websites from config
            for url in WEBSITE_LIST:
                try:
                    total_chunks, _ = process_website(url)
                    logger.info(f"Processed website {url} with {total_chunks} chunks")
                except Exception as e:
                    logger.error(f"Error processing website {url}: {str(e)}")
                
                # Add delay between processing websites
                time.sleep(5)
            
            # Process PDF files (existing directory watch logic)
            # ... existing directory watching code ...
            
            # Sleep before next iteration
            time.sleep(300)  # 5 minutes between checks
            
        except Exception as e:
            logger.error(f"Error in watch_directory_and_websites: {str(e)}")
            time.sleep(60)

if __name__ == "__main__":
    INPUT_DIR = "input_pdfs"
    PROCESSED_DIR = "processed_pdfs"
    ERROR_DIR = "error_pdfs"
    
    logger.info("Starting PDF and website processor service")
    watch_directory_and_websites(INPUT_DIR, PROCESSED_DIR, ERROR_DIR) 