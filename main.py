import asyncio

from starlette.responses import FileResponse
from typing import List
from fastapi import FastAPI, UploadFile, Request, File, Response, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from io import BytesIO
import re
import os
import shutil
from pdf2image import convert_from_path

from ocr.tesseract import read_image
from preprocessing.preprocessor import Preprocessor, ImagePipeline
from quality_metrics.text_metrics_report import TextMetricsReport
import logging

# Create a logger
logger = logging.getLogger(__name__)

# Set global level of logger. Can be overridden by individual handlers by setting their levels
logger.setLevel(logging.INFO)

# Create a file handler
handler = logging.FileHandler('logs/app.log')
handler.setLevel(logging.INFO)

# Create a logging format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

# Add the handlers to the logger
logger.addHandler(handler)

app = FastAPI(debug=True)
logging.basicConfig(level=logging.INFO)
app.mount("/static", StaticFiles(directory="frontend"), name="static")
frontend = Jinja2Templates(directory="frontend")
path = "temp"


@app.get("/")
def root(request: Request):
    return frontend.TemplateResponse("index.html", {"request": request})


@app.post("/convert")
async def convert(images: List[UploadFile] = File(...)):
    if len(images) == 0:
        raise HTTPException(status_code=400, detail="No images provided")
    results = []
    for image in images:
        filename = _store_file(image, image.filename)
        if image.content_type == 'application/pdf':
            images_from_pdf = convert_from_path(filename)
            image_path = filename[:-4] + ".jpeg"
            images_from_pdf[0].save(image_path, 'JPEG')
            filename = image_path

        # Preprocess the image
        preprocessor = Preprocessor(filename)
        pipeline = ImagePipeline(preprocessor)
        processed_image = pipeline.process_image()

        text = await read_image(processed_image, 'deu')
        os.remove(filename)
        file_name = image.filename.split('.')[0] + '.txt'
        file_bytes = BytesIO(text.encode())
        results.append(Response(content=file_bytes.getvalue(), media_type="text/plain",
                                headers={"Content-Disposition": f"attachment;filename={file_name}"}))
    return results


@app.post("/evaluation")
async def text_metrics_report(ocr_files: List[UploadFile] = File(...), gt_files: List[UploadFile] = File(...)):
    logger.info('Received request for text metrics report')
    if len(ocr_files) == 0 or len(gt_files) == 0:
        logger.warning('No text files provided')
        raise HTTPException(status_code=400, detail="No text files provided")

    def get_file_number(filename):
        # Split at the dot to separate the base from the extension
        base, _ = os.path.splitext(filename)
        # Get the trailing numbers using a regular expression
        match = re.search(r'\d+$', base)
        # Return the matched numbers or None if no numbers were found
        return match.group() if match else None

    ocr_files_dict = {get_file_number(ocr_file.filename): ocr_file for ocr_file in ocr_files}
    gt_files_dict = {get_file_number(gt_file.filename): gt_file for gt_file in gt_files}

    for file_number in ocr_files_dict.keys():
        if file_number not in gt_files_dict:
            logger.error(f"No matching ground truth file found for OCR file number {file_number}")
            raise HTTPException(status_code=400,
                                detail=f"No matching ground truth file found for OCR file number {file_number}")
    ocr_texts = []
    ground_truths = []
    filenames = []
    for filename, ocr_file in ocr_files_dict.items():
        gt_file = gt_files_dict[filename]

        # Store the file locally for processing
        file_to_process = _store_file(ocr_file, ocr_file.filename)
        if ocr_file.content_type == 'application/pdf':
            images_from_pdf = convert_from_path(file_to_process)
            image_path = file_to_process[:-4] + ".jpeg"
            images_from_pdf[0].save(image_path, 'JPEG')
            file_to_process = image_path

        # Preprocess the image
        preprocessor = Preprocessor(file_to_process)
        pipeline = ImagePipeline(preprocessor)
        processed_image = pipeline.process_image()

        # Perform OCR and store the result
        ocr_text = await read_image(processed_image, 'deu')

        # Read the ground truth text
        gt_text = await gt_file.read()

        ocr_texts.append(ocr_text)
        ground_truths.append(gt_text.decode('utf-8'))
        filenames.append(filename)

        # Delete the temporary file
        os.remove(file_to_process)

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
