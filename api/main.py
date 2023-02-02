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
    filename = _store_file(image)
    return {"filename": filename, "text": "this works"}


@app.post("/bulk_ocr")
async def bulk_ocr(request:Request, bgt: BackgroundTasks):
    return {"filename": "filename", "text": "this works"}



def _store_file(file):
    extension = os.path.splitext(file.filename)[-1]
    temp_file = os.path.join("../temp", "temp" + extension)
    with open(temp_file, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return temp_file



