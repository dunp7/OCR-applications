from fastapi import FastAPI, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
from pdf2image import convert_from_path
import pytesseract
import os
import shutil
from fastapi.responses import StreamingResponse
from io import BytesIO
from PIL import ImageDraw


# Initialize the FastAPI app
app = FastAPI()

# Directory for temporary file storage
TEMP_DIR = "./temp"
os.makedirs(TEMP_DIR, exist_ok=True)


pytesseract.pytesseract.tesseract_cmd = r"E:/extension/tesseract-ocr/tesseract.exe"


@app.post("/process_contract")
async def process_contract(file: UploadFile):
    """
    Endpoint to process a PDF file, extract text using OCR, and classify it into JSON structure.
    """
    # Ensure the uploaded file is a PDF
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    # Save the uploaded file to the temp directory
    file_path = os.path.join(TEMP_DIR, file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        # Convert PDF to images
        images = convert_from_path(file_path, poppler_path= 'E:/extension/poppler-24.07.0/Library/bin')
        
        # Extract text from each page and build the JSON structure
        document_data = {"document_classification": []}
        for page_number, image in enumerate(images, start=1):
            text = pytesseract.image_to_string(image, lang="eng")

            # Classify document type (basic example: keyword-based)
            document_data["document_classification"].append({
                "document_type": text,
                "page_numbers": [page_number]
            })

        # Return the JSON response
        return JSONResponse(content=document_data)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

    finally:
        # Clean up the temp file
        if os.path.exists(file_path):
            os.remove(file_path)





def extract_words_with_boxes(image):
    data = pytesseract.image_to_data(image, lang="eng", output_type=pytesseract.Output.DICT)
    words_info = []
    for i in range(len(data['text'])):
        word = data['text'][i].strip()
        conf = int(data['conf'][i])
        if word and conf > 0:
            words_info.append({
                "word": word,
                "confidence": conf,
                "left": data['left'][i],
                "top": data['top'][i],
                "width": data['width'][i],
                "height": data['height'][i]
            })
    return words_info
@app.post("/extract_words")
async def extract_words_with_boxes(file: UploadFile, page_number: int = Query(..., ge=1)):
    """
    Return the image of the specified PDF page with bounding boxes drawn around detected words.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    file_path = os.path.join(TEMP_DIR, file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        images = convert_from_path(file_path, poppler_path='E:/extension/poppler-24.07.0/Library/bin')
        if page_number > len(images):
            raise HTTPException(status_code=400, detail=f"PDF only has {len(images)} pages.")

        image = images[page_number - 1].convert("RGB")
        words_info = extract_words_with_boxes(image)

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


# Run the app with: uvicorn filename:app --reload
