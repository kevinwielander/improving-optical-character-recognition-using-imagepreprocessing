FROM python:3.9-buster


RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    libleptonica-dev \
    pkg-config \
    tesseract-ocr-deu \
    libgl1-mesa-glx

WORKDIR /improving-optical-character-recognition-using-imagepreprocessing

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8080

CMD ["uvicorn", "main:app", "--reload", "--host", "0.0.0.0", "--port", "8080"]
