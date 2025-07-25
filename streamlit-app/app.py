import streamlit as st
import logging
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
import time
import json
import re
from config.settings import MAX_WORKERS, SUPPORTED_LANGUAGES, MINIO_BUCKET_NAME, GOOGLE_API_KEY
from utils.ocr_utils import perform_ocr, convert_pdf_to_image, extract_words_with_boxes, draw_bounding_boxes
from utils.file_utils import save_uploaded_file, cleanup_file, split_pdf_by_titles, connect_to_minio, list_folders
from utils.api_utils import process_title, gen_answer


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize MinIO client
client = connect_to_minio()

# Initialize session state
if "document_titles" not in st.session_state:
    st.session_state.document_titles = []
if "file_path" not in st.session_state:
    st.session_state.file_path = None

# Main App
st.title("PDF OCR Extraction and Analysis")

# Sidebar
st.sidebar.header("Input Parameters")
lang = st.sidebar.selectbox("Language", SUPPORTED_LANGUAGES, index=0)
uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type=["pdf","xlsx",'docx'])
page_number = st.sidebar.number_input("Page Number", min_value=1, value=1, step=1)
worker_nums = st.sidebar.number_input("Number of Workers", min_value=1, max_value=MAX_WORKERS-2, value=1, step=1)
if st.sidebar.button("Clear Cache"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.sidebar.success("Cache cleared!")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["Extract Words", "Extract Raw Text", "Identify Title of Document", "Arrange Document to Folders"])

# Tab 1: Extract Words with Bounding Boxes
with tab1:
    st.header("Extract Words with Bounding Boxes")
    if st.button("Run Extract Words"):
        if not uploaded_file:
            st.error("Please upload a PDF file.")
        elif uploaded_file.name.endswith(".pdf"):
            file_path = save_uploaded_file(uploaded_file)
            try:
                image = convert_pdf_to_image(file_path, first_page=page_number, last_page=page_number)
                image = image[0].convert("RGB")
                words_info = extract_words_with_boxes(image, lang=lang)
                img_bytes = draw_bounding_boxes(image, words_info)
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
                image = convert_pdf_to_image(file_path, first_page=page_number, last_page=page_number)
                image = image[0].convert("RGB")
                raw_text = perform_ocr(image, lang=lang)
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
            folder_lists = list_folders(client, MINIO_BUCKET_NAME)
            try:
                with st.spinner("Processing..."):
                    images = convert_pdf_to_image(file_path, last_page=page_number)
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
                            future = executor.submit(process_title, page_count, text, prev_title, lang, prev_folder, folder_lists)
                            futures.append(future)
                            prev_title = future.result()[1]['title']
                            prev_folder = future.result()[1]['folder_recommendation']
                        page_results = [future.result() for future in futures]
                    st.info(f"API processing took {time.time() - start_time:.2f} seconds")

                    # Sort results by page number
                    page_results.sort(key=lambda x: x[0])

                    for page_count, title_obj in page_results:
                        raw_title = re.sub(r'\s+', ' ', title_obj.get("title", "")).strip()
                        raw_title = re.sub(r'[.,;:!]+$', '', raw_title).strip()
                        folder_rec = title_obj.get("folder_recommendation", "Other")

                        # Clean title
                        for char in ['\n', '.']:
                            raw_title = raw_title.replace(char, ' ')
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

                    document_titles = [
                        {
                            "title": title,
                            "page_numbers": data["pages"],
                            "folder_recommendation": data["folder"]
                        }
                        for title, data in title_pages_map.items()
                    ]
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

# Tab 4: Arrange Document to Folders
with tab4:
    st.header("Arrange Document to Folders")
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
        folders = st.text_input("Enter all folders name separated by comma", key="folder_name", value=",".join(folderlist))
        if st.button("Run Arrange Documents"):
            if folders:
                folders = [f.strip() for f in folders.replace(" ", "").split(",") if f.strip()]
            else:
                st.error("The folder name is blank -> Use AI Recommendation")
            logger.info(f"Arrange to these folders: {folders}")
            try:
                with st.spinner("Organizing PDFs..."):
                    if len(folders) == len(folderlist):
                        title_to_folder = {title["title"]: title["folder_recommendation"] for title in st.session_state.document_titles}
                    else:
                        titles = [doc["title"] for doc in st.session_state.document_titles]
                        prompt = f"""
                            Bạn là chuyên gia về sắp xếp các file vào folder theo tiêu đề.
                            Đây là danh sách các folder:
                            {json.dumps(folders)}
                            Đây là danh sách tiêu đề cần được sắp xếp:
                            {json.dumps(titles)}
                            Hãy tạo một JSON object ánh xạ mỗi tiêu đề đến một folder trong danh sách folder. 
                            Ưu tiên folder có từ khóa hoặc cụm từ khớp chính xác với tiêu đề (sau khi chuẩn hóa).
                            Nếu không tìm thấy folder phù hợp cho tiêu đề, gán nó vào folder "Other".
                            Đảm bảo mọi tiêu đề đều được ánh xạ.
                            Trả về chỉ JSON object, không giải thích.
                            Ví dụ:
                            {{"Title1": "Folder1", "Title2": "Folder2", "Unknown": "Other"}}
                        """
                        answer = gen_answer(logic_prompt=prompt, key=GOOGLE_API_KEY)
                        if "```json" in answer:
                            answer = answer.split("```json")[1].split("```")[0].strip()
                        elif "```" in answer:
                            answer = answer.split("```")[1].split("```")[0].strip()
                        try:
                            title_to_folder = json.loads(answer)
                        except json.JSONDecodeError:
                            st.error("Error parsing response from API.")
                            st.stop()
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
                    folder_structure = {}
                    for doc in st.session_state.document_titles:
                        title = doc["title"]
                        pages = doc["page_numbers"]
                        folder = title_to_folder.get(title, "Other")
                        if folder not in folder_structure:
                            folder_structure[folder] = []
                        folder_structure[folder].append({"title": title, "pages": pages})
                    st.subheader("Document Arrangement - Folder Structure")
                    for folder, docs in folder_structure.items():
                        with st.expander(f"📁 {folder}"):
                            for doc in docs:
                                st.markdown(f"📜 **{doc['title']}** (Pages: {', '.join(map(str, doc['pages']))})")
            except Exception as e:
                st.error(f"Error organizing PDFs: {str(e)}")
            finally:
                cleanup_file(file_path)