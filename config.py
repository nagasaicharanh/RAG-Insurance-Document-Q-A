import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Model Settings
MODEL_NAME = "llama-3.3-70b-versatile"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Chunking Settings
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# Retrieval Settings
TOP_K = 5

# Evaluation Thresholds
FAITHFULNESS_THRESHOLD = 0.85
CONTEXT_PRECISION_THRESHOLD = 0.80

# Paths
DATA_DIR = "data"
RAW_PDF_DIR = os.path.join(DATA_DIR, "raw_pdfs")
CHROMA_DB_DIR = "chroma_db"
TEST_QA_PATH = os.path.join(DATA_DIR, "test_qa.json")

# Ensure directories exist
os.makedirs(RAW_PDF_DIR, exist_ok=True)
os.makedirs(CHROMA_DB_DIR, exist_ok=True)