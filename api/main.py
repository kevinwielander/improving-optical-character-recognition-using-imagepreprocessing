from typing import List
from fastapi import FastAPI, UploadFile, Request, File, BackgroundTasks
from fastapi.templating import Jinja2Templates
import cv2
import numpy as np
import os
import shutil

app = FastAPI()
frontend = Jinja2Templates(directory="../frontend")


@app.get("/")
def root(request: Request):
    return frontend.TemplateResponse("index.html", {"request":request})

@app.post("/single_file_ocr")
async def single_file_ocr(image: UploadFile = File(...)):
    path = "../temp/single"
    os.mkdir(path)
    filename = _store_file(image, path, image.filename)
    #os.remove(path)
    return {"filename": filename, "text": "this works"}


@app.post("/bulk_ocr")
async def bulk_ocr(images: List[UploadFile] = File(...)):
    path = "../temp/bulk"
    os.mkdir(path)
    for image in images:
        _store_file(image,path,image.filename)
    #os.remove(path)
    return {"filename": "filename", "text": "this works"}



def _store_file(file, path, name):
    extension = os.path.splitext(file.filename)[-1]
    temp_file = os.path.join(path, name + extension)
    with open(temp_file, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return temp_file



