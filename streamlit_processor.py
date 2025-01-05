import streamlit as st
import os
from pathlib import Path
from datetime import datetime
from pdf_processor import process_pdf, logger

st.set_page_config(page_title="Phantom Directive - PDF Processor")

# Apply the same styling as the main app
st.markdown("""
<style>
    /* Dark theme with neon green accents */
    .stApp {
        background-color: #0a0a0a !important;
    }
    
    /* All text elements */
    .stApp, .stMarkdown, div[data-testid="stMarkdownContainer"] p {
        color: #00ff00 !important;
    }
    
    /* File uploader */
    .stUploadedFile {
        background-color: #1a1a1a !important;
        border: 1px solid #00ff00 !important;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #1a1a1a !important;
        color: #00ff00 !important;
        border: 1px solid #00ff00 !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("PDF Processor")

# Create directories if they don't exist
for dir_name in ['input_pdfs', 'processed_pdfs', 'error_pdfs', 'logs']:
    Path(dir_name).mkdir(exist_ok=True)

# File uploader
uploaded_files = st.file_uploader("Upload PDF files", type=['pdf'], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        st.write(f"Processing: {uploaded_file.name}")
        
        try:
            # Get PDF bytes
            pdf_bytes = uploaded_file.read()
            
            # Process PDF
            namespace = f"{datetime.now().strftime('%Y%m%d')}-{uploaded_file.name}"
            with st.spinner("Processing..."):
                total_chunks, _ = process_pdf(pdf_bytes, uploaded_file.name, namespace)
            
            if total_chunks > 0:
                st.success(f"Successfully processed {uploaded_file.name}: {total_chunks} chunks created")
                
                # Save to processed directory
                with open(f"processed_pdfs/{uploaded_file.name}", "wb") as f:
                    f.write(pdf_bytes)
            else:
                st.error(f"Failed to process {uploaded_file.name}")
                
                # Save to error directory
                with open(f"error_pdfs/{uploaded_file.name}", "wb") as f:
                    f.write(pdf_bytes)
                
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {str(e)}")
            
            # Save to error directory
            with open(f"error_pdfs/{uploaded_file.name}", "wb") as f:
                f.write(pdf_bytes)

# Show processing history
st.markdown("---")
st.subheader("Processing History")

# Show recent logs
try:
    with open('logs/processor.log', 'r') as f:
        logs = f.readlines()[-20:]  # Show last 20 lines
        for log in logs:
            st.text(log.strip())
except FileNotFoundError:
    st.text("No processing history yet.") 