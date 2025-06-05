from fastapi import FastAPI, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
from pdf2image import convert_from_path
import pytesseract
import os
import shutil
from fastapi.responses import StreamingResponse
from io import BytesIO
from PIL import ImageDraw
import openai
from utils import extract_words_with_boxes, gen_answer
import json
import tiktoken

# Initialize the FastAPI app
app = FastAPI()

# Directory for temporary file storage
TEMP_DIR = "./temp"
os.makedirs(TEMP_DIR, exist_ok=True)

pytesseract.pytesseract.tesseract_cmd = r"error_fix/tesseract-ocr/tesseract.exe"  

#----------------------------------------General--------------------------------------------------------
@app.post("/extract_words_per_page", tags = ['General'])
async def extract_words(file: UploadFile, lang: str = "vie",page_number: int = Query(..., ge=1)):
    """
    Return the image of the specified PDF page with bounding boxes drawn around detected words.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    file_path = os.path.join(TEMP_DIR, file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        
        images = convert_from_path(file_path, poppler_path='error_fix/poppler-24.07.0/Library/bin')  #In Windows System Only
        # images = convert_from_path(file_path)

        if page_number > len(images):
            raise HTTPException(status_code=400, detail=f"PDF only has {len(images)} pages.")

        image = images[page_number - 1].convert("RGB")
        words_info = extract_words_with_boxes(image, lang=lang)

        draw = ImageDraw.Draw(image)
        for w in words_info:
            left = w['left']
            top = w['top']
            right = left + w['width']
            bottom = top + w['height']
            draw.rectangle([left, top, right, bottom], outline="red", width=2)

        # Save image to bytes
        img_bytes = BytesIO()
        image.save(img_bytes, format="PNG")
        img_bytes.seek(0)

        return StreamingResponse(img_bytes, media_type="image/png")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

#----------------------------------------Menu--------------------------------------------------------
SAMPLE_SCHEMA = {
    "products": [
        {
            "title": "Shrimp with Broccoli",
            "price": "14.45",
            "category": "Seafood",
            "is_active": True,
            "main_ingredients": ["Shrimp", "Broccoli"],
        },
        {
            "title": "Shrimp with Chinese Vegetables",
            "price": "14.45",
            "category": "Seafood",
            "is_active": True,
            "main_ingredients": ["Shrimp", "Chinese Vegetables"],
        },
    ],
}

@app.post("/extract_raw_text_per_page", tags = ['Menu'])
async def extract_raw_text(file: UploadFile, lang: str = "vie", page_number: int = Query(..., ge=1)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    file_path = os.path.join(TEMP_DIR, file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    try:
        
        images = convert_from_path(file_path, poppler_path='error_fix/poppler-24.07.0/Library/bin')  #In Windows System Only
        # images = convert_from_path(file_path)

        if page_number > len(images):
            raise HTTPException(status_code=400, detail=f"PDF only has {len(images)} pages.")

        image = images[page_number - 1].convert("RGB")
        raw_text = pytesseract.image_to_string(image, lang=lang)

        return {"page_number": page_number, "raw_text": raw_text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")
    
@app.post("/extract_order_per_page", tags = ['Menu'])
async def extract_menu(file: UploadFile, lang: str = "vie", page_number: int = Query(..., ge=1)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    file_path = os.path.join(TEMP_DIR, file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        
        images = convert_from_path(file_path, poppler_path='error_fix/poppler-24.07.0/Library/bin')  #In Windows System Only
        # images = convert_from_path(file_path)

        if page_number > len(images):
            raise HTTPException(status_code=400, detail=f"PDF only has {len(images)} pages.")

        image = images[page_number - 1].convert("RGB")
        # raw_text = pytesseract.image_to_string(image, lang=lang)
        words_info = extract_words_with_boxes(image, lang=lang)
        word_list = [f"{line['word']}|{line['left']},{line['top']},{line['width']},{line['height']}" for line in words_info]
        ocr_text_with_pos = "\n".join(word_list)

        client = openai.OpenAI(api_key= OPENAI_KEY)
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system",
                 "content":(
                    "You are an assistant for extracting structured menu data from OCR text. "
                    "Your task is to return a JSON object adhering to the following schema: "
                    f"{json.dumps(SAMPLE_SCHEMA)}. "
                    "Only use the data from the provided OCR text and position. "
                    "Each row is each word with position: word|left,top,width,height. "
                    "If some information is missing, use placeholders or null values. "
                    "Do not include explanations or extra text in the output."
                    f"All field values in the JSON object must be {lang} language.")},
                {
                "role": "user", "content": f"OCR text with positions: {ocr_text_with_pos}"},
            ],
            temperature=0.2,
            max_tokens=2500)
        try:
            menu_extract = json.loads(response.choices[0].message.content.strip())
            return {"status": "success", "token": response.usage.total_tokens ,"data": menu_extract}
        except Exception as e:
            return {"status": "error in json load"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

#----------------------------------------contract--------------------------------------------------------
@app.post("/extract_title_per_page", tags = ['Contract'])
async def extract_title(file: UploadFile, lang: str = "vie", page_number: int = Query(..., ge=1), api_key: str = Query(...)):
    """
    Extract and return the text of the specified PDF page.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    file_path = os.path.join(TEMP_DIR, file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        images = convert_from_path(file_path, poppler_path='E:/extension/poppler-24.07.0/Library/bin') #In Window System Only
        # images = convert_from_path(file_path)
        if page_number > len(images):
            raise HTTPException(status_code=400, detail=f"PDF only has {len(images)} pages.")

        image = images[page_number - 1].convert("RGB")
        # Use Tesseract to extract text from the image
        text = pytesseract.image_to_string(image, lang=lang)

        # extract title
        prompt = f"Xác định tiêu đề của văn bản sau: {text}. Nếu như không có thì trả về None"
        title = gen_answer(prompt, api_key)


        # Return the extracted text
        return {"page_number": page_number, 'title': title}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


@app.post("/classify_document", tags = ['Contract'])
async def classify_document(file: UploadFile, api_key: str = Query(...), lang: str = "vie"):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    file_path = os.path.join(TEMP_DIR, file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        images = convert_from_path(file_path, poppler_path='E:/extension/poppler-24.07.0/Library/bin') #In Window System Only
        images = convert_from_path(file_path)
        title_pages_map = {}
        title_list = []

        for page_number, image in enumerate(images, start=1):
            text = pytesseract.image_to_string(image, lang= lang)

            if title_list:
                prev_title = title_list[-1]
            else:
                prev_title = None

            prompt = (
                (f"Trang trước có tiêu đề: {prev_title}.\n" if prev_title else "")
                + f"Đoạn văn bản sau đây:\n{text}\n"
                "Văn bản này có thuộc tiêu đề trên không? "
                "Nếu có, chỉ trả về đúng tiêu đề đó. "
                "Nếu không thuộc, hãy tìm tiêu đề mới phù hợp nhất dựa trên nội dung này. "
                "Nếu không xác định được, hãy trả về tiêu đề của trang trước. "
                "Nếu không có tiêu đề nào trước đó, hãy tìm tiêu đề phù hợp cho đoạn văn bản này"
                "Chỉ trả về chính xác tên tiêu đề hoặc None, không thêm giải thích hay thông tin nào khác."
            )

            title = gen_answer(prompt, api_key).strip()

            if not title or title.lower() in ["none", "null", "unknown", ""]:
                title = "Unknown"

            if title in title_pages_map:
                title_pages_map[title].append(page_number)
            else:
                title_pages_map[title] = [page_number]
                title_list.append(title)

        document_titles = [
            {"title": t, "page_numbers": pages}
            for t, pages in title_pages_map.items()
        ]

        return JSONResponse(content={"document_titles": document_titles})

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

