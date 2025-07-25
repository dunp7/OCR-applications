import os
from pathlib import Path
from dotenv import load_dotenv

# Project root directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
load_dotenv()

# Environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TESSERACT_PATH = os.path.join(PROJECT_ROOT, os.getenv("TESSERACT_FIX"))
POPPLER_PATH = os.path.join(PROJECT_ROOT, os.getenv("POPPLER_FIX"))


# Constants
TEMP_DIR = "./temp"
MAX_WORKERS = os.cpu_count()
SUPPORTED_LANGUAGES = ["vie", "eng"]


# Minio DB:
MINIO_URL= "10.16.91.164:2004"
MINIO_BUCKET_NAME= "project-ocr"
MINIO_USERNAME = "minio"
MINIO_PASSWORD = "minio12345"