import streamlit as st
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
import re
import time
import PyPDF2
import logging
from utils.ocr_utils import perform_ocr, convert_pdf_to_image, list_xlsx_sheets, sheet_to_md, convert_docx
from utils.api_utils import process_title
from utils.file_utils import cleanup_file
logger = logging.getLogger(__name__)
## -------------------------------------------Tab 3: Identify Title of Document
def clean_title(raw_title, title_list):
    """Clean title and handle invalid cases."""
    raw_title = re.sub(r'\s+', ' ', raw_title.strip())
    raw_title = re.sub(r'[.,;:!]+$', '', raw_title).strip()
    for char in ['\n', '.', ',', ':']:
        raw_title = raw_title.replace(char, ' ')
    if not raw_title or raw_title.lower() in ["none", "null", "unknown", "không xác định"]:
        return title_list[-1] if title_list else "Không Xác Định", "Other"
    return raw_title, None

def map_titles(page_results, title_pages_map, title_list, log_interval=10):
    """Map page/sheet results to document titles and log progress."""
    page_results.sort(key=lambda x: x[0])
    for page_idx, title_obj in page_results:
        raw_title = title_obj.get("title", "")
        folder_rec = title_obj.get("folder_recommendation", "Other")
        cleaned_title, default_folder = clean_title(raw_title, title_list)
        if default_folder:
            folder_rec = default_folder

        if cleaned_title in title_pages_map:
            title_pages_map[cleaned_title]["pages"].append(page_idx)
        else:
            title_pages_map[cleaned_title] = {"pages": [page_idx], "folder": folder_rec}
            title_list.append(cleaned_title)

        if page_idx % log_interval == 0:
            logger.info(f"Processed {'page' if isinstance(page_idx, int) else 'sheet'} {page_idx}")

    return [
        {"title": title, "page_numbers/ sheet numbers": data["pages"], "folder_recommendation": data["folder"]}
        for title, data in title_pages_map.items()
    ]

def process_pdf(file_path, page_number, worker_nums, lang, folder_lists, do_ocr=True):
    """Process PDF file to extract titles."""
    with st.spinner("Processing..."):
        if do_ocr == True:
            st.info("Performing OCR...")
            start_time = time.time()
            images = convert_pdf_to_image(file_path, last_page=page_number)
            with ProcessPoolExecutor(max_workers=worker_nums) as executor:
                text_results = list(executor.map(partial(perform_ocr, lang=lang), images))
            st.info(f"OCR processing took {time.time() - start_time:.2f} seconds")
        else: 
            start_time = time.time()
            # Extract text from PDF without OCR
            with open(file_path, 'rb') as file:
                pdf = PyPDF2.PdfReader(file)
                text_results = [page.extract_text() or "" for page in pdf.pages[:page_number]]
            st.info(f"Direct text extraction took {time.time() - start_time:.2f} seconds")

        page_results = []
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=worker_nums) as executor:
            futures = []
            prev_title = None
            prev_folder = None
            for page_count, text in enumerate(text_results, start=1):
                future = executor.submit(process_title, page_count, text, prev_title, lang, prev_folder, folder_lists)
                futures.append(future)
                prev_title = future.result()[1]['title']
                prev_folder = future.result()[1]['folder_recommendation']
            page_results = [future.result() for future in futures]
        st.info(f"API processing took {time.time() - start_time:.2f} seconds")

    title_pages_map, title_list = {}, []
    return map_titles(page_results, title_pages_map, title_list)

def process_xlsx(file_path, page_number, worker_nums, lang, folder_lists):
    """Process XLSX file to extract titles."""
    with st.spinner("Processing..."):
        sheet_names = list_xlsx_sheets(file_path)[:page_number]
        start_time = time.time()
        sheet_texts = [str(sheet_to_md(file_path, sheet_name=sheet)) for sheet in sheet_names]
        st.info(f"Excel sheet processing took {time.time() - start_time:.2f} seconds")

        page_results = []
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=worker_nums) as executor:
            futures = []
            prev_title = None
            prev_folder = None
            for sheet_idx, text in enumerate(sheet_texts, start=1):
                future = executor.submit(process_title, sheet_idx, text, prev_title, lang, prev_folder, folder_lists)
                futures.append(future)
                prev_title = future.result()[1]['title']
                prev_folder = future.result()[1]['folder_recommendation']
            page_results = [future.result() for future in futures]
        st.info(f"API processing took {time.time() - start_time:.2f} seconds")

    title_pages_map, title_list = {}, []
    return map_titles(page_results, title_pages_map, title_list)


def process_docx(file_path, page_number, worker_nums, lang, folder_lists, do_ocr):
    """Process DOCX file by converting to PDF and extracting titles from half the pages."""
    with st.spinner("Converting DOCX to PDF..."):
        pdf_path = convert_docx(file_path)

    try:    
        result = process_pdf(pdf_path, page_number, worker_nums, lang, folder_lists, do_ocr)
    except Exception as e:
        print(e)
    finally:
        print(f"File {pdf_path} deleted")
        cleanup_file(pdf_path)
    return result