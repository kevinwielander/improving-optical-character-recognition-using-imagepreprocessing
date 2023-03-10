import asyncio
from typing import List
from fastapi import FastAPI, UploadFile, Request, File, Response, HTTPException
from fastapi.templating import Jinja2Templates
from io import BytesIO
import os
import shutil
from PIL import Image
from pdf2image import convert_from_path

from ocr.tesseract import read_image
from preprocessing.preprocessor import Preprocessor

app = FastAPI()
frontend = Jinja2Templates(directory="frontend")
path = "temp"

@app.get("/")
def root(request: Request):
    return frontend.TemplateResponse("index.html", {"request": request})


@app.post("/single_file_ocr")
async def single_file_ocr(image: UploadFile = File(...)):
    filename = _store_file(image, image.filename)
    if image.content_type == 'application/pdf':
        images = convert_from_path(filename)
        image_path = filename[:-4] + ".jpeg"
        images[0].save(image_path, 'JPEG')
        filename = image_path

    # Preprocess the image
    preprocessor = Preprocessor(filename)
    preprocessor.to_grayscale()
    preprocessor.check_and_scale_dpi()

    text = await read_image(filename, 'deu')
    os.remove(filename)
    file_name = image.filename.split('.')[0] + '.txt'
    file_bytes = BytesIO(text.encode())
    return Response(content=file_bytes.getvalue(), media_type="text/plain",
                    headers={"Content-Disposition": f"attachment;filename={file_name}"})


@app.post("/bulk_ocr")
async def bulk_ocr(images: List[UploadFile] = File(...)):
    if len(images) == 0:
        raise HTTPException(status_code=400, detail="No images provided")
    results = []
    for image in images:
        filename = _store_file(image, image.filename)
        text = await read_image(filename, 'deu')
        os.remove(filename)
        file_name = filename.split('.')[0] + '.txt'
        file_bytes = BytesIO(text.encode())
        results.append(Response(content=file_bytes.getvalue(), media_type="text/plain",
                                headers={"Content-Disposition": f"attachment;filename={file_name}"}))
    return results


def _store_file(file, name):
    temp_file = os.path.join(path, name)
    with open(temp_file, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return temp_file



