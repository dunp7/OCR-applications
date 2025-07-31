import json
import re
from pydantic import BaseModel
from config.settings import GOOGLE_API_KEY
from google import genai
from google.genai.types import GenerateContentConfig
import time
class AnswerFormat(BaseModel):
    title: str
    folder_recommendation: str

def gen_answer(logic_prompt, key, system_prompt= None, config = None):
    client = genai.Client(api_key= key)
    if not config:
        config = GenerateContentConfig(
            temperature=0,
            top_k=1,
            top_p=1,
            system_instruction=system_prompt)
        
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=logic_prompt,
        config=config
    ).text
    time.sleep(3)
    return response

def process_title(page_count, text, prev_title, lang, prev_folder, folderlist):
    """Process title for a single page."""
    prompt = f"""
    Bạn là chuyên gia xác định tiêu đề văn bản bằng ngôn ngữ {lang}. Nhiệm vụ:
    1. Xác định tiêu đề chính xác, đảm bảo ngữ pháp chuẩn và chuyên nghiệp. Nếu không xác định được, trả về "Unknown".
    2. Đề xuất folder phù hợp nhất từ danh sách folder hoặc tạo folder mới nếu cần.

    ### Thông tin tham khảo:
    - Tiêu đề trang trước: {prev_title if prev_title else "Không có"}.
    - Folder trang trước: {prev_folder if prev_folder else "Không có"}.
    - Danh sách folder: {json.dumps(folderlist, ensure_ascii=False) if folderlist else "Không có folder nào"}.
    - Văn bản: \n{text}\n

    ### Quy tắc:
    **1. Tiêu đề:**
    - Loại bỏ dấu xuống dòng, khoảng trắng thừa, và số tài liệu.
    - Sửa lỗi chính tả phổ biến.
    - Nếu tiêu đề trang trước vẫn phù hợp với nội dung hiện tại → dùng lại `prev_title`.
    - Nếu không, xác định tiêu đề mới từ nội dung.
    - Nếu không thể xác định được, trả về `"Unknown"`.

    **2. Folder:**
    - Nếu dùng lại tiêu đề 'prev_title' của trang trước → dùng lại `prev_folder` của trang trước.
    - Nếu tiêu đề mới:
        + Ưu tiên folder khớp chính xác với tiêu đề trong danh sách folder(nếu có) (so sánh sau khi chuẩn hóa).
        + Nếu không có folder khớp, đề xuất **tên folder mới** từ nội dung hoặc tiêu đề.
        + folder mới cần ngắn gọn, nếu lên dự án mà nó thuộc về. 

    ### Đầu ra:
    - `title`: Tiêu đề (giữ định dạng gốc, không có số tài liệu, khoảng trắng thừa) hoặc "Unknown".
    - `folder_recommendation`: Tên folder phù hợp nhất hoặc folder mới đề xuất, hoặc "Unknown" nếu không xác định được.
    Không thêm giải thích hoặc ký tự thừa.
    """
    print(prompt)
    config = GenerateContentConfig(
        temperature=0,
        top_k=1,
        top_p=1,
        response_mime_type="application/json",
        response_schema=AnswerFormat
    )
    try:
        title = gen_answer(prompt, GOOGLE_API_KEY, config=config)
        title = json.loads(title)
        return page_count, title
    except Exception as e:
        return page_count, {"title": "Unknown", "folder_recommendation": "Unknown"}