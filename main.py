
import itertools
from starlette.responses import FileResponse
from typing import List
from fastapi import FastAPI, UploadFile, Request, File, Response, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from io import BytesIO
from datetime import datetime
import re
import os
import shutil
from pdf2image import convert_from_path

from ocr.tesseract import read_image
from preprocessing.preprocessor import Preprocessor, ImagePipeline
from quality_metrics.text_metrics import TextMetrics
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
    logger.info('Received request for index.html')
    return frontend.TemplateResponse("index.html", {"request": request})


@app.post("/convert")
async def convert(images: List[UploadFile] = File(...)):
    logger.info('Received request for convert Endpoint')
    if len(images) == 0:
        logger.error('No images provided')
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
        processing_steps = ['filter', 'non_local_means', 'morphological_operation', 'thresholding']
        processed_image = pipeline.process_image(processing_steps)

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
        processing_steps = ['filter', 'non_local_means', 'morphological_operation', 'thresholding']
        processed_image = pipeline.process_image(processing_steps)

        # Perform OCR and store the result
        ocr_text = await read_image(processed_image, 'deu')

        # Read the ground truth text
        gt_text = await gt_file.read()

        ocr_texts.append(ocr_text)
        ground_truths.append(gt_text.decode('utf-8'))
        filenames.append(filename)

        # Delete the temporary file
        os.remove(file_to_process)

    report = TextMetricsReport(ground_truths, ocr_texts, filenames, processing_steps)
    report.generate_report()

    for ocr_file, gt_file in zip(ocr_files, gt_files):
        await ocr_file.close()
        await gt_file.close()

    return FileResponse(report.filename, media_type='text/csv',
                        headers={"Content-Disposition": f"attachment;filename={report.filename}"})


@app.post("/experiment")
async def experiment(ocr_files: List[UploadFile] = File(...), gt_files: List[UploadFile] = File(...)):
    logger.info('Received request for OCR experiment')
    if len(ocr_files) == 0 or len(gt_files) == 0:
        logger.warning('No text files provided')
        raise HTTPException(status_code=400, detail="No text files provided")

    def get_file_number(filename):
        base, _ = os.path.splitext(filename)
        match = re.search(r'\d+$', base)
        return match.group() if match else None

    ocr_files_dict = {get_file_number(ocr_file.filename): ocr_file for ocr_file in ocr_files}
    gt_files_dict = {}
    for gt_file in gt_files:
        content = await gt_file.read()
        content = content.decode('utf-8')
        gt_files_dict[get_file_number(gt_file.filename)] = content

    for file_number in ocr_files_dict.keys():
        if file_number not in gt_files_dict:
            logger.error(f"No matching ground truth file found for OCR file number {file_number}")
            raise HTTPException(status_code=400,
                                detail=f"No matching ground truth file found for OCR file number {file_number}")

    preprocess_methods = ['filter', 'non_local_means', 'morphological_operation', 'thresholding']
    combinations = list(itertools.chain(*map(lambda x: itertools.combinations(preprocess_methods, x),
                                             range(0, len(preprocess_methods) + 1))))

    all_metrics = []
    i = 0
    total_iterations = len(combinations) * len(ocr_files_dict)
    start_time = datetime.now()
    for combo in combinations:
        for filename, ocr_file in ocr_files_dict.items():
            preprocess_steps = []
            original_file = _store_file(ocr_file, ocr_file.filename)
            image_path = original_file[:-5] + ".jpeg"
            if ocr_file.content_type == 'application/pdf':
                if not os.path.exists(image_path):  # Checking if the image already exists
                    images_from_pdf = convert_from_path(original_file)
                    images_from_pdf[0].save(image_path, 'JPEG')
            file_to_process = image_path

            preprocessor = Preprocessor(file_to_process)
            image_pipeline = ImagePipeline(preprocessor)

            # Apply selected preprocessing methods via pipeline
            processed_image = image_pipeline.process_image(combo)

            ocr_text = await read_image(processed_image, 'deu')
            gt_text = gt_files_dict[filename]

            preprocess_steps.append(' '.join(str(step) for step in combo) if len(combo) > 0 else 'No preprocessing')

            tm = TextMetrics(gt_text, ocr_text)
            wer = tm.wer()
            cer = tm.cer()
            lev_distance = tm.lev_distance()

            all_metrics.append(
                {'Index': i, 'Filename': filename, 'Preprocessing Steps': ', '.join(map(str, preprocess_steps)),
                 'WER': wer, 'CER': cer,
                 'Levenshtein Distance': lev_distance})
            i += 1
            # print progress
            if i % 10 == 0:
                current_time = datetime.now()
                elapsed_time = current_time - start_time
                avg_time_per_iteration = elapsed_time / i
                remaining_iterations = total_iterations - i
                estimated_end_time = current_time + avg_time_per_iteration * remaining_iterations
                logger.info(f'Progress: {i}/{total_iterations}, Elapsed Time: {elapsed_time},'
                            f' Estimated End Time: {estimated_end_time}')

    for ocr_file, gt_file in zip(ocr_files, gt_files):
        await ocr_file.close()
        await gt_file.close()

    logging.info(all_metrics)
    report = TextMetricsReport(all_metrics=all_metrics)
    report.write_to_csv()

    return FileResponse(report.filename, media_type='text/csv',
                        headers={"Content-Disposition": f"attachment;filename={report.filename}"})


@app.post('/process_results')
async def process_csv(request: Request):
    form = await request.form()
    csv_file = form["file"]
    metric = form["metric"]  # Get selected metric

    filename = _store_file(csv_file, csv_file.filename)
    report = TextMetricsReport()
    result_filename = report.analyze_experiment(filename, metric)

    # Return the file
    return FileResponse(result_filename, media_type='text/csv')



def _store_file(file, name):
    temp_file = os.path.join(path, name)
    # Check if the file already exists
    if not os.path.isfile(temp_file):
        with open(temp_file, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    return temp_file


def copy_file(original_path, new_path):
    shutil.copy2(original_path, new_path)
