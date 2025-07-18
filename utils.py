import pytesseract
from google import genai
from google.genai.types import GenerateContentConfig
import time

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



def gen_answer(logic_prompt, key, system_prompt= None):
    client = genai.Client(api_key= key)
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=logic_prompt,
        config=GenerateContentConfig(
            temperature=0,
            top_k=1,
            top_p=1,
            system_instruction=system_prompt)
    ).text
    time.sleep(4)
    return response