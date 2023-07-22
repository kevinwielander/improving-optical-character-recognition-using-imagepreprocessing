FROM python:3.9-alpine

RUN apk update && \
    apk add tesseract-ocr tesseract-ocr-dev leptonica-dev g++ make libc-dev tesseract-ocr-data-deu


WORKDIR /improving-optical-character-recognition-using-imagepreprocessing

EXPOSE 8080

COPY requirements.txt .
COPY venv venv

RUN pip install -r requirements.txt

COPY . .

RUN . venv/Scripts/activate

CMD ["uvicorn", "main:app", "--reload", "--host", "0.0.0.0", "--port", "8080"]
