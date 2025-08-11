import pytesseract
from pdf2image import convert_from_path
from PIL import Image, ImageDraw
from io import BytesIO
from config.settings import POPPLER_PATH, TESSERACT_PATH
import pandas as pd
from docx2pdf import convert
import subprocess
import os
pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH # Windows


# --- XLSX SHEET
def list_xlsx_sheets(file_path):
    try: 
        with pd.ExcelFile(file_path) as xls:
            return xls.sheet_names
    except Exception as e:
        print(e)
        return []


def sheet_to_md(file_path, sheet_name):
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name, engine="openpyxl")
        # Drop empty rows and columns
        df = df.dropna(how='all').dropna(axis=1, how='all')
        if df.empty:
            return "⚠️ The selected sheet is empty or contains only NaN values."
        return df.to_markdown(index=False, tablefmt="github")
    except Exception as e:  
        return f"❌ Error reading sheet '{sheet_name}': {str(e)}"



#--- DOCX TEXT
def convert_docx(file_path):
    try: 
        pdf_path = file_path.replace(".docx", ".pdf")
        print(pdf_path)
        print(file_path)
        convert(file_path, pdf_path)
        return pdf_path
    except Exception as e:
        print(e)
        return None

# def convert_docx(input_path: str):
#     output_dir = os.path.dirname(input_path)
#     try:
#         subprocess.run([
#             "soffice",
#             "--headless",
#             "--convert-to", "pdf",
#             "--outdir", output_dir,
#             input_path
#         ], check=True)

#         pdf_path = input_path.replace(".docx", ".pdf")
#         if os.path.exists(pdf_path):
#             return pdf_path
#         else:
#             print("❌ PDF not created")
#             return None
#     except Exception as e:
#         print(f"❌ Error converting DOCX to PDF: {e}")
#         return None

# --- PDF OCR
def perform_ocr(image, lang):
    """Perform OCR on an image and return extracted text."""
    return pytesseract.image_to_string(image, lang=lang)

def extract_words_with_boxes(image, lang="eng"):
    data = pytesseract.image_to_data(image, lang= lang, output_type=pytesseract.Output.DICT)
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

def convert_pdf_to_image(file_path, first_page=1, last_page=1):
    """Convert a PDF page to an image."""
    try:
        images = convert_from_path(file_path, poppler_path=POPPLER_PATH, first_page=first_page, last_page=last_page) # In Windows
        # images = convert_from_path(file_path, first_page=first_page, last_page=last_page)
        if not isinstance(images, list):
            images = [images]
        return images
    except Exception as e:
        raise Exception(f"Error converting PDF to image: {str(e)}")

def draw_bounding_boxes(image, words_info):
    """Draw bounding boxes on the image and return as BytesIO."""
    draw = ImageDraw.Draw(image)
    for w in words_info:
        left = w['left']
        top = w['top']
        right = left + w['width']
        bottom = top + w['height']
        draw.rectangle([left, top, right, bottom], outline="red", width=2)
    img_bytes = BytesIO()
    image.save(img_bytes, format="PNG")
    img_bytes.seek(0)
    return img_bytes

if __name__ == "__main__":
    pass