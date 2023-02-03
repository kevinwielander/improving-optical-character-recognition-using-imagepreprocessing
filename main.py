from typing import List
from fastapi import FastAPI, UploadFile, Request, File
from fastapi.templating import Jinja2Templates
import os
import shutil

from ocr.tesseract import read_image

app = FastAPI()
frontend = Jinja2Templates(directory="frontend")


@app.get("/")
def root(request: Request):
    return frontend.TemplateResponse("index.html", {"request": request})


@app.post("/single_file_ocr")
async def single_file_ocr(image: UploadFile = File(...)):
    path = "temp"
    filename = _store_file(image, path, image.filename)
    text = await read_image(filename, 'eng')
    os.remove(filename)
    return {"filename": filename, "text": text}


@app.post("/bulk_ocr")
async def bulk_ocr(images: List[UploadFile] = File(...)):
    path = "../temp/bulk"
    os.mkdir(path)
    for image in images:
        _store_file(image, path, image.filename)
    #os.remove(path)
    return {"filename": "filename", "text": "this works"}



def _store_file(file, path, name):
    temp_file = os.path.join(path, name)
    with open(temp_file, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return temp_file



