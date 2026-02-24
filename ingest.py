import os
import glob
import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import config
from vector_store import add_documents, is_document_ingested

def parse_pdf(file_path):
    doc = fitz.open(file_path)
    documents = []
    source_name = os.path.basename(file_path)
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text()
        
        if text.strip():
            metadata = {
                "source": source_name,
                "page": page_num + 1,
                "doc_type": "reinsurance"
            }
            documents.append(Document(page_content=text, metadata=metadata))
            
    return documents

def chunk_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )
    return text_splitter.split_documents(documents)

def main():
    pdf_files = glob.glob(os.path.join(config.RAW_PDF_DIR, "*.pdf"))
    
    if not pdf_files:
        print(f"No PDFs found in {config.RAW_PDF_DIR}")
        return
        
    for file_path in pdf_files:
        source_name = os.path.basename(file_path)
        if is_document_ingested(source_name):
            print(f"Skipping {source_name}: already ingested.")
            continue
            
        print(f"Processing {source_name}...")
        documents = parse_pdf(file_path)
        chunks = chunk_documents(documents)
        print(f"  Generated {len(chunks)} chunks.")
        
        add_documents(chunks)
        print(f"  Successfully ingested {source_name}.")

if __name__ == "__main__":
    main()