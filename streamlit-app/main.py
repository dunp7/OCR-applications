import streamlit as st
from pdf2image import convert_from_path
import pytesseract
from PIL import Image, ImageDraw
from io import BytesIO
import os
import shutil
import json
import logging
from dotenv import load_dotenv
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
import time
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)
from utils import extract_words_with_boxes, gen_answer
from google.genai.types import GenerateContentConfig
from pydantic import BaseModel
from file_system import connect_to_minio, list_folders, split_pdf_by_titles
import re


class AnswerFormat(BaseModel):
    title: str
    folder_recommendation: str

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TESSERACT_PATH = os.path.join(PROJECT_ROOT, os.getenv("TESSERACT_FIX"))
POPPLER_PATH = os.path.join(PROJECT_ROOT, os.getenv("POPPLER_FIX"))
MINIO_BUCKET_NAME= "project-ocr"

# CONFIG
max_workers = os.cpu_count()
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Directory for temporary file storage
TEMP_DIR = "./temp"
os.makedirs(TEMP_DIR, exist_ok=True)

# Set Tesseract path (Windows-specific, adjust as needed)
pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH # Windows

# Set cache for variables
if "document_titles" not in st.session_state:
    st.session_state.document_titles = []
if "file_path" not in st.session_state:
    st.session_state.filepaths = None

# MAIN APP
st.title("PDF OCR Extraction and Analysis")

st.sidebar.header("Input Parameters")
lang = st.sidebar.selectbox("Language", ["vie", "eng"], index=0)
uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type=["pdf"])
page_number = st.sidebar.number_input("Page Number", min_value=1, value=1, step=1)
worker_nums = st.sidebar.number_input("Number of Workers", min_value=1, max_value=max_workers-2,value=1, step=1)
if st.sidebar.button("Clear Cache"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.sidebar.write("Cache cleared!")

# Function to save uploaded file
def save_uploaded_file(uploaded_file):
    file_path = os.path.join(TEMP_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path


def cleanup_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)

def perform_ocr(image, lang):
    return pytesseract.image_to_string(image, lang=lang)

# Function to process title for a single page
def process_title(page_count, text, prev_title, lang, prev_folder, folderlist):
    prompt = f"""
        Bạn là một chuyên gia về xác định tiêu đề của văn bản của ngôn ngữ {lang}. 
        Nhiệm vụ của bạn là:
        1. xác định tiêu đề chính xác, đảm bảo ngữ pháp {lang} chuẩn mực và chuyên nghiệp. Hãy xác định chính xác tiêu đề của văn bản, nếu không xác định được, trả về None.
        2. Chọn folder phù hợp nhất từ danh sách folder dựa trên tiêu đề và nội dung văn bản.
        ### Tài liệu tham khảo:
        Trang trước có tiêu đề: {prev_title if prev_title else ""}.\n", thuộc folder {prev_folder if prev_folder else ""}.
        -----------------------------
        Đoạn văn bản sau đây:\n{text}\n"
        -----------------------------
        Đây là danh sách các folder có sẵn: \n{folderlist}\n

        ### Quy tắc:
        1. Loại bỏ dấu xuống dòng (\\n), khoảng trắng thừa, và số tài liệu khỏi văn bản trước khi xác định tiêu đề.
        2. Sửa lỗi chính tả thường gặp trong ngôn ngữ {lang} để đảm bảo tiêu đề đúng ngữ pháp.
        3. Chuẩn hóa tiêu đề bằng cách chuyển về chữ hoa để so sánh (nhưng giữ nguyên định dạng gốc khi trả về).
        4. Nếu tiêu đề chuẩn hóa khớp với tiêu đề chuẩn hóa của trang trước, trả về tiêu đề gốc của trang trước.
        5. Nếu không khớp, xác định tiêu đề mới dựa trên nội dung văn bản (sau khi loại bỏ số tài liệu).
        6. Nếu không thể xác định tiêu đề, trả về tiêu đề gốc của trang trước (nếu có) hoặc "Unknown" nếu không có tiêu đề trước.
        7. Kiểm tra chính tả của tiêu đề trước khi trả về.
        8. Để chọn folder:
        - Ưu tiên folder có từ khóa hoặc cụm từ khớp chính xác với tiêu đề (sau khi chuẩn hóa).
        - Nếu không có folder khớp chính xác, chọn folder có từ khóa gần nhất với nội dung chính của văn bản (dựa trên ngữ nghĩa hoặc từ khóa chính).
        - Nếu không tìm được folder phù hợp, trả về "Unknown" cho folder.
        ### yêu cầu đầu ra:
        - `tiêu_đề`: Tên tiêu đề (giữ định dạng gốc, không bao gồm số tài liệu, không có dấu xuống dòng hoặc khoảng trắng thừa) hoặc "Unknown".
        - `folder_khớp`: Tên folder từ `folderlist` phù hợp nhất với tiêu đề hoặc nội dung văn bản, hoặc "Unknown" nếu không tìm thấy folder phù hợp.
        - Không thêm giải thích hay ký tự thừa.
    """
    # logger.info(prompt)
    config = GenerateContentConfig(
                temperature=0,
                top_k=1,
                top_p=1,
                response_mime_type="application/json",
                response_schema=AnswerFormat)
    title = gen_answer(prompt, GOOGLE_API_KEY, config= config)

    try: 
        title = json.loads(title)
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return page_count, ["Unknown", "Unknown"]
    return page_count, title, 

# Tabs for different functionalities
tab1, tab2, tab3, tab4= st.tabs(["Extract Words", "Extract Raw Text", "Identify Title of Document", "Arrange Document to Folders"])

# Tab 1: Extract Words with Bounding Boxes
with tab1:
    st.header("Extract Words with Bounding Boxes")
    if st.button("Run Extract Words"):
        if not uploaded_file:
            st.error("Please upload a PDF file.")
        elif uploaded_file.name.endswith(".pdf"):
            file_path = save_uploaded_file(uploaded_file)
            try:
                images = convert_from_path(file_path, poppler_path=POPPLER_PATH, first_page=page_number, last_page=page_number) # In Windows
                # images = convert_from_path(file_path, first_page=page_number, last_page=page_number)
                image = images[0].convert("RGB")
                words_info = extract_words_with_boxes(image, lang=lang)
                draw = ImageDraw.Draw(image)
                for w in words_info:
                    left = w['left']
                    top = w['top']
                    right = left + w['width']
                    bottom = top + w['height']
                    draw.rectangle([left, top, right, bottom], outline="red", width=2)
                img_bytes = BytesIO()
                image.save(img_bytes, format="PNG")
                st.image(img_bytes, caption=f"Page {page_number} with Bounding Boxes")


                del images, image
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
            finally:
                cleanup_file(file_path)
        else:
            st.error("Only PDF files are allowed.")

# Tab 2: Extract Raw Text
with tab2:
    st.header("Extract Raw Text")
    if st.button("Run Extract Raw Text"):
        if not uploaded_file:
            st.error("Please upload a PDF file.")
        elif uploaded_file.name.endswith(".pdf"):
            file_path = save_uploaded_file(uploaded_file)
            try:
                images = convert_from_path(file_path, poppler_path=POPPLER_PATH, first_page=page_number, last_page=page_number) # In Windows
                # images = convert_from_path(file_path, first_page=page_number, last_page=page_number)
                image = images[0].convert("RGB")
                raw_text = pytesseract.image_to_string(image, lang=lang)
                st.write(f"**Page Number**: {page_number}")
                st.text_area("Raw Text", raw_text, height=300)

                del images, image
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
            finally:
                cleanup_file(file_path)
        else:
            st.error("Only PDF files are allowed.")


# Tab 3: Identify Title of Document
with tab3:
    st.header("Identify Title of Document")
    if st.button("Run Classify Document"):
        if not uploaded_file:
            st.error("Please upload a PDF file.")
            
        elif uploaded_file.name.endswith(".pdf"):
            file_path = save_uploaded_file(uploaded_file)
            client = connect_to_minio()
            folder_lists = list_folders(client, MINIO_BUCKET_NAME)
            try:
                with st.spinner("Processing..."):
                    images = convert_from_path(file_path, poppler_path=POPPLER_PATH, first_page=1, last_page=page_number)
                    # images = convert_from_path(file_path, first_page=1, last_page=page_number)
                    title_pages_map = {}
                    title_list = []

                    # Parallel OCR processing
                    start_time = time.time()
                    with ProcessPoolExecutor(max_workers=worker_nums) as executor:
                        ocr_results = list(executor.map(partial(perform_ocr, lang=lang), images))
                    st.info(f"OCR processing took {time.time() - start_time:.2f} seconds")

                    # Parallel API calls for title extraction
                    page_results = []
                    start_time = time.time()
                    with ThreadPoolExecutor(max_workers=worker_nums) as executor:
                        
                        futures = []
                        prev_title = None
                        prev_folder = None
                        for page_count, text in enumerate(ocr_results, start=1):
                            future = executor.submit(process_title, page_count, text, prev_title, lang, prev_folder ,folder_lists)
                            futures.append(future)
                            prev_title = future.result()[1]['title']
                            prev_folder = future.result()[1]['folder_recommendation']
                        page_results = [future.result() for future in futures]
                        

                    st.info(f"API processing took {time.time() - start_time:.2f} seconds")

                    # Sort results by page number to maintain order
                    page_results.sort(key=lambda x: x[0])
                    
                    for page_count, title_obj in page_results:
                        raw_title = re.sub(r'\s+', ' ', title_obj.get("title", "") )
                        raw_title = re.sub(r'[.,;:!]+$', '', title_obj.get("title", "") ).strip()
                        folder_rec = title_obj.get("folder_recommendation", "Other")

                        # Làm sạch title
                        char = ['\n', '.']
                        for c in char:
                            raw_title = raw_title.replace(c, ' ')

                        if not raw_title or raw_title.lower() in ["none", "null", "unknown", "không xác định"]:
                            raw_title = title_list[-1] if title_list else "Không Xác Định"
                            folder_rec = "Other"

                        if raw_title in title_pages_map:
                            title_pages_map[raw_title]["pages"].append(page_count)
                        else:
                            title_pages_map[raw_title] = {
                                "pages": [page_count],
                                "folder": folder_rec
                            }
                            title_list.append(raw_title)

                        if page_count % 10 == 0:
                            logger.info(f"Processed page {page_count}")
                document_titles = [{
                                    "title": title,
                                    "page_numbers": data["pages"],
                                    "folder_recommendation": data["folder"]
                                    }
                                    for title, data in title_pages_map.items()]
                st.session_state.document_titles = document_titles
                if document_titles:
                    st.json({"document_titles": document_titles})
                else:
                    st.error("No document titles found.")

            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
            finally:
                cleanup_file(file_path)
        else:
            st.error("Only PDF files are allowed.")


# Tab 4: Display Document Titles
with tab4:
    st.header("Arrange Document to Folders")

    # Check folder name
    client = connect_to_minio()
    folderlist = list_folders(client, MINIO_BUCKET_NAME)
    st.subheader("📂 All Folders in Bucket")
    if folderlist:
        for folder in folderlist:
            st.markdown(f"- 📁 **{folder}**")
    else:
        st.info("No folders found in the bucket.")
    st.subheader("Arrange Folders")
    if not st.session_state.document_titles:
        st.error("No document titles found. Please run 'Identify Title of Document' first.")
    else:
        folders = st.text_input("Enter all folders name separated by comma", key="folder_name" ,value= ",".join(folderlist))
        if st.button("Run Arrange Documents"):
            if not folders:
                st.error("Please enter all folders name separated by comma.")
            folders = [f.strip() for f in folders.replace(" ", "").split(",") if f.strip()]
            logger.info(f"Arrange to these folders: {folders}")
            try: 
                with st.spinner("Organizing PDFs..."):
                    if len(folders) == len(folderlist):
                        title_to_folder = {title["title"]: title["folder_recommendation"] for title in st.session_state.document_titles} 
                    else:
                        # Prepare prompt for mapping titles to folders
                        titles = [doc["title"] for doc in st.session_state.document_titles]
                        prompt = f"""
                            Bạn là chuyên gia về sắp xếp các file vào folder theo tiêu đề.

                            </inference material>
                            Đây là danh sách các folder:
                            {json.dumps(folders)}

                            Đây là danh sách tiêu đề cần được sắp xếp:
                            {json.dumps(titles)}
                            <inference material>

                            </rule>
                            Hãy tạo một JSON object ánh xạ mỗi tiêu đề đến một folder trong danh sách folder. 
                            Ưu tiên folder có từ khóa hoặc cụm từ khớp chính xác với tiêu đề (sau khi chuẩn hóa).
                            Nếu không tìm thấy folder phù hợp cho tiêu đề, gán nó vào folder "Other".
                            Đảm bảo mọi tiêu đề đều được ánh xạ.
                            <rule>
                            
                            </output format>
                            Trả về chỉ JSON object, không giải thích.

                            Ví dụ:
                            '{{
                                "Title1": "Folder1",
                                "Title2": "Folder2",
                                "Unknown": "Other"
                            }}'
                            <output format>
                            """
                        logger.info(f"Prompt: {prompt}")
                        answer = gen_answer(logic_prompt=prompt, key=GOOGLE_API_KEY)
                        if "```json" in answer:
                            answer = answer.split("```json")[1].split("```")[0].strip()
                        elif "```" in answer:
                            answer = answer.split("```")[1].split("```")[0].strip()
                        # st.write(answer)
                        try:
                            title_to_folder = json.loads(answer)
                        except json.JSONDecodeError:
                            st.error("Error parsing response from API.")
                            st.stop()
                        
                        # Validate that all titles are mapped
                        missing_titles = [t for t in titles if t not in title_to_folder]
                        if missing_titles:
                            st.warning(f"Some titles were not mapped: {missing_titles}. They will be placed in 'Other'.")
                            for t in missing_titles:
                                title_to_folder[t] = "Other"
                    file_path = save_uploaded_file(uploaded_file)
                    uploaded = split_pdf_by_titles(
                        input_pdf_path=file_path,
                        document_titles=st.session_state.document_titles,
                        client=client,
                        bucket_name=MINIO_BUCKET_NAME
                    )
                    # Create folder structure
                    folder_structure = {}
                    for doc in st.session_state.document_titles:
                        title = doc["title"]
                        pages = doc["page_numbers"]
                        folder = title_to_folder.get(title, "Other")
                        if folder not in folder_structure:
                            folder_structure[folder] = []
                        folder_structure[folder].append({"title": title, "pages": pages})
                    # Display folder structure
                    st.subheader("Document Arrangement - Folder Structure")
                    for folder, docs in folder_structure.items():
                        with st.expander(f"📁 {folder}"):
                            for doc in docs:
                                st.markdown(f"📜 **{doc['title']}** (Pages: {', '.join(map(str, doc['pages']))})")
            except Exception as e:
                st.error(f"Error organizing PDFs: {str(e)}")
            finally:
                cleanup_file(file_path)