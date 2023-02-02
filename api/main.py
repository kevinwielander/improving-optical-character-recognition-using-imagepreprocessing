from typing import List
from fastapi import FastAPI, UploadFile, Request
from fastapi.templating import Jinja2Templates
import cv2
import numpy as np

app = FastAPI()
frontend = Jinja2Templates(directory="../frontend")


@app.get("/")
def root(request: Request):
    return frontend.TemplateResponse("index.html", {"request":request})

@app.post("/preprocess")
async def preprocess(file: UploadFile):
    img = cv2.imdecode(np.frombuffer(await file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
     
     # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
   
    # Encode the grayscale image data into a byte string
    encoded_img = cv2.imencode('.jpg', gray)[1].tobytes()

    # Return the encoded grayscale image data as binary data
    return {"image": encoded_img, "file_name": file.filename}

@app.post("/ocr")
async def ocr(files: List[UploadFile]):
    recognized_texts = []
    
    for file in files:
        # Load the image from the file
        img = cv2.imread(file.file.name)
        
        # Set the Tesseract data path from the environment variable
        # pytesseract.pytesseract.tesseract_cmd = os.environ.get("TESSERACT_DATA_PATH", "tesseract")
        
        # Use Tesseract to perform OCR on the image
        text = "test return text"
        
        recognized_texts.append(text)
    
    return {"recognized_texts": recognized_texts}
