import streamlit as st
from pdf2image import convert_from_path
import pytesseract
from PIL import Image, ImageDraw
from io import BytesIO
import os
import shutil
import openai
import json
import logging
from utils import extract_words_with_boxes, gen_answer
from dotenv import load_dotenv
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TESSERACT_PATH = os.getenv("TESSERACT_FIX")
POPPLER_PATH = os.getenv("POPPLER_FIX")


# CONFIG
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Directory for temporary file storage
TEMP_DIR = "./temp"
os.makedirs(TEMP_DIR, exist_ok=True)

# Set Tesseract path (Windows-specific, adjust as needed)
pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

# Set cache for variables
if "document_titles" not in st.session_state:
    st.session_state.document_titles = []


# MAIN APP
st.title("PDF OCR Extraction and Analysis")

st.sidebar.header("Input Parameters")
lang = st.sidebar.selectbox("Language", ["vie", "eng"], index=0)
uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type=["pdf"])
page_number = st.sidebar.number_input("Page Number", min_value=1, value=1, step=1)


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
                images = convert_from_path(file_path, poppler_path=POPPLER_PATH)
                if page_number > len(images):
                    st.error(f"PDF only has {len(images)} pages.")
                else:
                    image = images[page_number - 1].convert("RGB")
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
                images = convert_from_path(file_path, poppler_path='error_fix/poppler-24.07.0/Library/bin')
                if page_number > len(images):
                    st.error(f"PDF only has {len(images)} pages.")
                else:
                    image = images[page_number - 1].convert("RGB")
                    raw_text = pytesseract.image_to_string(image, lang=lang)
                    st.write(f"**Page Number**: {page_number}")
                    st.text_area("Raw Text", raw_text, height=300)
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
            try:
                with st.spinner("Processing..."):
                    images = convert_from_path(file_path, poppler_path='error_fix/poppler-24.07.0/Library/bin', first_page=1, last_page=page_number)
                    title_pages_map = {}
                    title_list = []
                    for page_count, image in enumerate(images, start=1):
                        text = pytesseract.image_to_string(image, lang=lang)
                        prev_title = title_list[-1] if title_list else None

                        prompt = f"""
                            Bạn là một chuyên gia về xác định tiêu đề của văn bản của ngôn ngữ {lang}. 
                            Nhiệm vụ của bạn là xác định tiêu đề chính xác, đảm bảo ngữ pháp {lang} chuẩn mực và chuyên nghiệp. 
                            Hãy xác định chính xác tiêu đề của văn bản, nếu không xác định được, trả về None.
                            </inference material>
                            Trang trước có tiêu đề: {prev_title if prev_title else ""}.\n"
                            -----------------------------
                            Đoạn văn bản sau đây:\n{text}\n"
                            <inference material>

                            </rule>
                            -Loại bỏ tất cả dấu xuống dòng (\\n), khoảng trắng thừa, và các số tài liệu khỏi văn bản trước khi xác định tiêu đề.
                            -Sửa các lỗi chính tả của ngôn ngữ {lang} thường gặp để đảm bảo tiêu đề đúng ngữ pháp.
                            -Chuẩn hóa tiêu đề bằng cách chuyển về chữ hoa để so sánh (nhưng giữ nguyên định dạng gốc khi trả về).
                            -Nếu tiêu đề chuẩn hóa khớp với tiêu đề chuẩn hóa của trang trước, trả về tiêu đề gốc của trang trước.
                            -Nếu không khớp, xác định tiêu đề mới phù hợp nhất dựa trên nội dung văn bản (sau khi loại bỏ số tài liệu).
                            -Nếu không thể xác định tiêu đề, trả về tiêu đề gốc của trang trước (nếu có) hoặc "Unknown" nếu không có tiêu đề trước.
                            -Kiểm tra lại chính tả của câu trả lời 
                            <rule>
                            </output format>
                            -Chỉ trả về tên tiêu đề (giữ định dạng gốc, không bao gồm số tài liệu, không có dấu xuống dòng hoặc khoảng trắng thừa) hoặc "Unknown", không thêm giải thích hay ký tự thừa.
                            <output format>
                        """
                        # logger.info(f"Prompt: {prompt}")
                        title = gen_answer(prompt, GOOGLE_API_KEY)
                        
                        char = ['\n', '.']
                        for c in char:
                            title = title.replace(c, ' ')
                        title = title.strip()
                        if not title or title.lower() in ["none", "null", "unknown", ""]:
                            title = "Unknown"
                        if title in title_pages_map:
                            title_pages_map[title].append(page_count)
                        else:
                            title_pages_map[title] = [page_count]
                            title_list.append(title)
                        if page_count % 10 == 0:
                            logger.info(f"Processed page {page_count}")
                document_titles = [{"title": t, "page_numbers": pages} for t, pages in title_pages_map.items()]
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
    if not st.session_state.document_titles:
        st.error("No document titles found. Please run 'Identify Title of Document' first.")
    else:
        folders = st.text_input("Enter all folders name separated by comma", key="folder_name")
        if st.button("Run Arrange Documents"):
            if not folders:
                st.error("Please enter all folders name separated by comma.")
            folders = [f.strip() for f in folders.replace(" ", "").split(",") if f.strip()]
            logger.info(f"Arrange to these folders: {folders}")
            try: 
                with st.spinner("Organizing PDFs..."):
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
                    
                    # Display results
                    st.subheader("Document Arrangement Results")
                    for doc in st.session_state.document_titles:
                        title = doc["title"]
                        folder = title_to_folder.get(title, "Other")
                        pages = doc["page_numbers"]
                        st.write(f"**Title**: {title}")
                        st.write(f"**Assigned Folder**: {folder}")
                        st.write(f"**Page Numbers**: {', '.join(map(str, pages))}")
                        st.write("---")
            except Exception as e:
                st.error(f"Error organizing PDFs: {str(e)}")