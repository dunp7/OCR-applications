FROM python:3.12-slim


WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    poppler-utils \
    tesseract-ocr \
    libleptonica-dev \
    libtesseract-dev \
    python3-pil \
    tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

COPY . /app


RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]