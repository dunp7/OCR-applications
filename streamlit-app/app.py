import streamlit as st
import logging
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
import time
import json
import re
from config.settings import MAX_WORKERS, SUPPORTED_LANGUAGES, MINIO_BUCKET_NAME, GOOGLE_API_KEY
from utils.ocr_utils import perform_ocr, convert_pdf_to_image, extract_words_with_boxes, draw_bounding_boxes
from utils.ocr_utils import list_xlsx_sheets, sheet_to_md, convert_docx
from utils.file_utils import save_uploaded_file, cleanup_file, split_pdf_by_titles, connect_to_minio, list_folders, split_xlsx_by_titles
from utils.api_utils import gen_answer
from utils.app_utils import process_pdf, process_xlsx, process_docx
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize MinIO client
client = connect_to_minio()

# Initialize session state
if "document_titles" not in st.session_state:
    st.session_state.document_titles = []


# Main App
st.title("Text/ OCR Extraction and Analysis")

# Sidebar
st.sidebar.header("Input Parameters")
lang = st.sidebar.selectbox("Language", SUPPORTED_LANGUAGES, index=0)
uploaded_file = st.sidebar.file_uploader("Upload a file", type=["pdf","xlsx",'docx'])
page_number = st.sidebar.number_input("Page/ Sheet Number", min_value=1, value=1, step=1)
worker_nums = st.sidebar.number_input("Number of Workers", min_value=1, max_value=MAX_WORKERS-2, value=1, step=1)
do_ocr = st.sidebar.checkbox("Perform OCR on all pages (Applied with PDF, DOCX)", value=True)
if st.sidebar.button("Clear Cache"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.sidebar.success("Cache cleared!")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["Extract Words", "Extract Raw Text", "Identify Title of Document", "Arrange Document to Folders"])

# Tab 1: Extract Words with Bounding Boxes
with tab1:
    st.header("Extract Words with Bounding Boxes Using OCR")
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
            st.error("Please upload a PDF/ XLSX/ DOCX file.")
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
        elif uploaded_file.name.endswith(".xlsx"):
            file_path = save_uploaded_file(uploaded_file)
            try:
                sheet_names = list_xlsx_sheets(file_path)
                st.write(f"**Sheet Name**: {sheet_names[page_number-1]}")

                md_text = sheet_to_md(file_path, sheet_name=sheet_names[page_number-1])
                st.markdown(md_text, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
            finally:
                cleanup_file(file_path)
        else:
            st.error("Not the right files are allowed.")

# Tab 3: Identify Title of Document
with tab3:
    st.header("Identify Title of Document")
    if st.button("Run Classify Document"):
        if not uploaded_file:
            st.error("Please upload a file.")
        else:
            file_path = save_uploaded_file(uploaded_file)
            folder_lists = list_folders(client, MINIO_BUCKET_NAME)
            file_type = uploaded_file.name.split(".")[-1]
            try:
                if file_type == "pdf":
                    document_titles = process_pdf(file_path, page_number, worker_nums, lang, folder_lists, do_ocr=do_ocr)
                elif file_type == "xlsx":
                    document_titles = process_xlsx(file_path, page_number, worker_nums, lang, folder_lists)
                elif file_type == "docx":
                    document_titles = process_docx(file_path, page_number, worker_nums, lang, folder_lists, do_ocr=do_ocr)
                else:
                    st.error("Only PDF, XLSX and DOCX files are allowed.")
                    document_titles = None

                if document_titles:
                    st.session_state.document_titles = document_titles
                    st.json({"document_titles": document_titles})
                elif not st.error("No document titles found."):  
                    pass
            except Exception as e:
                st.error(f"Error processing file {file_type}: {str(e)}")
            finally:
                print(f"File {file_path} deleted")
                cleanup_file(file_path)

# Tab 4: Arrange Document to Folders
with tab4:
    # Hiện folder hiện có
    st.header("Arrange Document to Folders")
    folderlist = list_folders(client, MINIO_BUCKET_NAME)
    st.subheader("📂 All Folders in Bucket")
    if folderlist:
        for folder in folderlist:
            st.markdown(f"- 📁 **{folder}**")
    else:
        st.info("No folders found in the bucket.")
    # Sắp xếp folder
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
                with st.spinner("Organizing documents..."):
                    if len(folders) == len(folderlist):
                        # ko điền folder name -> sử dụng AI recommendation
                        title_to_folder = {title["title"]: title["folder_recommendation"] for title in st.session_state.document_titles}
                    else:
                        # ko điền folder name -> sử dụng người dùng nhập
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
                                uploaded = split_xlsx_by_titles(
                                    input_xlsx_path=file_path,
                                    document_titles=st.session_state.document_titles,
                                    client=client,
                                    bucket_name=MINIO_BUCKET_NAME,
                                )
                    
                    
                    file_path = save_uploaded_file(uploaded_file)
                    file_type = uploaded_file.name.split(".")[-1].lower()
                    # Process based on file type
                    if file_type == "pdf":
                        print(file_path)
                        # Split PDF by titles and save to folders
                        uploaded = split_pdf_by_titles(
                            input_pdf_path=file_path,
                            document_titles=st.session_state.document_titles,
                            client=client,
                            bucket_name=MINIO_BUCKET_NAME,
                            input_type="pdf"
                        )
                    
                    elif file_type == "docx":
                        print(file_path)
                        pdf_file_path = convert_docx(file_path)
                        try:
                            cleanup_file(file_path)
                        except Exception as e:
                            print(e)
                        uploaded = split_pdf_by_titles(
                            input_pdf_path=pdf_file_path,
                            document_titles=st.session_state.document_titles,
                            client=client,
                            bucket_name=MINIO_BUCKET_NAME,
                            input_type="docx"
                        )

                    
                    elif file_type == "xlsx":
                        print(file_path)
                        uploaded = split_xlsx_by_titles(
                                input_xlsx_path=file_path,
                                document_titles=st.session_state.document_titles,
                                client=client,
                                bucket_name=MINIO_BUCKET_NAME
                        )
                    
                    else:
                        st.error("Only PDF, XLSX, and DOCX files are allowed.")
                        st.stop()

                    # Display folder structure
                    folder_structure = {}
                    for doc in st.session_state.document_titles:
                        title = doc["title"]
                        pages = doc["page_numbers/ sheet numbers"]
                        folder = title_to_folder.get(title, "Other")
                        if folder not in folder_structure:
                            folder_structure[folder] = []
                        folder_structure[folder].append({"title": title, "pages": pages})
                    st.subheader("Document Arrangement - Folder Structure")
                    for folder, docs in folder_structure.items():
                        with st.expander(f"📁 {folder}"):
                            for doc in docs:
                                pages_info = doc["pages"] if isinstance(doc["pages"], str) else ', '.join(map(str, doc["pages"]))
                                st.markdown(f"📜 **{doc['title']}** (Pages: {pages_info})")
            except Exception as e:
                st.error(f"Error organizing {file_type}: {str(e)}")
            finally:
                cleanup_file(file_path)