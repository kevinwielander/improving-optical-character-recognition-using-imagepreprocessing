import asyncio

from starlette.responses import FileResponse
from typing import List
from fastapi import FastAPI, UploadFile, Request, File, Response, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from io import BytesIO
import os
import shutil
import logging
from PIL import Image
from pdf2image import convert_from_path

from ocr.tesseract import read_image
from preprocessing.preprocessor import Preprocessor
from quality_metrics.text_metrics_report import TextMetricsReport

app = FastAPI()
logging.basicConfig(level=logging.INFO)
app.mount("/static", StaticFiles(directory="frontend"), name="static")
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


@app.post("/text_metrics")
async def text_metrics_report(ocr_files: List[UploadFile] = File(...), gt_files: List[UploadFile] = File(...)):
    if len(ocr_files) == 0 or len(gt_files) == 0:
        raise HTTPException(status_code=400, detail="No text files provided")

    ocr_files_dict = {ocr_file.filename: ocr_file for ocr_file in ocr_files}
    gt_files_dict = {gt_file.filename: gt_file for gt_file in gt_files}
    for file_name in ocr_files_dict.keys():
        if file_name not in gt_files_dict:
            raise HTTPException(status_code=400, detail=f"No matching ground truth file found for {file_name}")

    ocr_texts = []
    ground_truths = []
    filenames = []
    for filename, ocr_file in ocr_files_dict.items():
        gt_file = gt_files_dict[filename]

        ocr_text = await ocr_file.read()
        gt_text = await gt_file.read()

        ocr_texts.append(ocr_text.decode('utf-8'))
        ground_truths.append(gt_text.decode('utf-8'))
        filenames.append(filename)

    report = TextMetricsReport(ground_truths, ocr_texts, filenames)
    report.generate_report()

    for ocr_file, gt_file in zip(ocr_files, gt_files):
        await ocr_file.close()
        await gt_file.close()

    return FileResponse(report.filename, media_type='text/csv',
                        headers={"Content-Disposition": f"attachment;filename={report.filename}"})


def _store_file(file, name):
    temp_file = os.path.join(path, name)
    with open(temp_file, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return temp_file
