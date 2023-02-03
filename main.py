from typing import List
from fastapi import FastAPI, UploadFile, Request, File
from fastapi.templating import Jinja2Templates
import os
import shutil

from ocr.tesseract import read_image

app = FastAPI()
frontend = Jinja2Templates(directory="frontend")
path = "temp"

@app.get("/")
def root(request: Request):
    return frontend.TemplateResponse("index.html", {"request": request})


@app.post("/single_file_ocr")
async def single_file_ocr(image: UploadFile = File(...)):
    filename = _store_file(image, image.filename)
    text = await read_image(filename, 'eng')
    os.remove(filename)
    return {"filename": filename, "text": text}


@app.post("/bulk_ocr")
async def bulk_ocr(images: List[UploadFile] = File(...)):
    results = []
    for image in images:
        filename = _store_file(image, image.filename)
        text = await read_image(filename, 'eng')
        os.remove(filename)
        results.append({"filename": filename, "text": text})
    return results


def _store_file(file, name):
    temp_file = os.path.join(path, name)
    with open(temp_file, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return temp_file



