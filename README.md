# PDF Processing and Embedding Pipeline

This application provides a robust pipeline for processing PDF documents, extracting their content, and generating embeddings using state-of-the-art language models. It includes both a core processor and a Streamlit-based user interface for easy interaction.

## Features

- PDF document processing and text extraction
- Document embedding generation using OpenAI's text-embedding-3-small model
- Vector storage integration with Pinecone
- Error handling and logging
- Streamlit web interface for easy document management
- SQLite database for tracking processed files

## Prerequisites

- Python 3.8+
- Virtual environment (recommended)
- API keys for OpenAI, Anthropic, and Pinecone services

## Installation

1. Clone the repository
2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
3. Install dependencies:
```bash
pip install -r requirements.txt
```
4. Copy `.env.example` to `.env` and fill in your API keys and configuration:
```bash
cp .env.example .env
```

## Project Structure

- `processor.py`: Main processing pipeline
- `streamlit_processor.py`: Streamlit web interface
- `init_db.py`: Database initialization script
- `input_pdfs/`: Directory for incoming PDFs
- `processed_pdfs/`: Successfully processed PDFs
- `error_pdfs/`: PDFs that encountered processing errors
- `logs/`: Application logs
- `config/`: Configuration files

## Usage

1. Initialize the database:
```bash
python init_db.py
```

2. Run the Streamlit interface:
```bash
streamlit run streamlit_processor.py
```

3. Place PDF files in the `input_pdfs` directory for processing

## Environment Variables

Copy `.env.example` to `.env` and configure the following variables:

- `OPENAI_API_KEY`: Your OpenAI API key
- `OPENAI_EMBEDDING_MODEL`: Embedding model to use
- `ANTHROPIC_API_KEY`: Your Anthropic API key
- `ANTHROPIC_MODEL`: Anthropic model to use
- `PINECONE_API_KEY`: Your Pinecone API key
- `PINECONE_ENVIRONMENT`: Pinecone environment
- `PINECONE_INDEX_NAME`: Name of your Pinecone index

## License

This project is licensed under the MIT License - see the LICENSE file for details. 