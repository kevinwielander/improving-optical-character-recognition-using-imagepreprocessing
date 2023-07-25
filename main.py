import io
import itertools

import pandas as pd
from starlette.responses import FileResponse
from typing import List
from fastapi import FastAPI, UploadFile, Request, File, Response, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from io import BytesIO
from datetime import datetime
import os
from pdf2image import convert_from_path

from classification.feature_extraction import ImageFeaturesExtractor
from classification.train_model import PreprocessingOptimization
from ocr.tesseract import read_image
from preprocessing.image_pipeline import ImagePipeline
from quality_metrics.text_metrics import TextMetrics
from quality_metrics.text_metrics_report import TextMetricsReport
from utils.config import PREPROCESSING_STEPS, LOGS_PATH, LOG_FILE
from utils.helpers import prepare_file_dicts, store_file
import logging

# Create a logger
logger = logging.getLogger(__name__)

# Set global level of logger. Can be overridden by individual handlers by setting their levels
logger.setLevel(logging.INFO)

os.makedirs(LOGS_PATH, exist_ok=True)
# Create a file handler

handler = logging.FileHandler(os.path.join(LOGS_PATH, LOG_FILE))
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
        filename = store_file(image, image.filename)
        if image.content_type == 'application/pdf':
            images_from_pdf = convert_from_path(filename)
            image_path = filename[:-4] + ".jpeg"
            images_from_pdf[0].save(image_path, 'JPEG')
            filename = image_path

        # Preprocess the image
        text = await process_and_read_image(filename, PREPROCESSING_STEPS)

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

    ocr_files_dict, gt_files_dict = prepare_file_dicts(ocr_files, gt_files)

    ocr_texts = []
    ground_truths = []
    filenames = []
    for filename, ocr_file in ocr_files_dict.items():
        gt_text = gt_files_dict[filename]

        # Store the file locally for processing
        file_to_process = store_file(ocr_file, ocr_file.filename)
        if ocr_file.content_type == 'application/pdf':
            images_from_pdf = convert_from_path(file_to_process)
            image_path = file_to_process[:-4] + ".jpeg"
            images_from_pdf[0].save(image_path, 'JPEG')
            file_to_process = image_path

        # Preprocess the image and perform OCR
        try:
            # Preprocess the image and perform OCR
            ocr_text = await process_and_read_image(file_to_process, PREPROCESSING_STEPS)
        finally:
            # Delete the temporary file
            os.remove(file_to_process)

        ocr_texts.append(ocr_text)
        ground_truths.append(gt_text)
        filenames.append(filename)

    report = TextMetricsReport(ground_truths, ocr_texts, filenames, PREPROCESSING_STEPS)
    report.generate_report()

    for ocr_file, gt_file in zip(ocr_files, gt_files):
        await ocr_file.close()
        await gt_file.close()

    return FileResponse(report.filename, media_type='text/csv',
                        headers={"Content-Disposition": f"attachment;filename={report.filename}"})


@app.post("/experiment")
async def experiment(ocr_files: List[UploadFile] = File(...), gt_files: List[UploadFile] = File(...), experiment_type: str = "variation"):
    logger.info('Received request for OCR experiment')
    if len(ocr_files) == 0 or len(gt_files) == 0:
        logger.warning('No text files provided')
        raise HTTPException(status_code=400, detail="No text files provided")

    ocr_files_dict, gt_files_dict = prepare_file_dicts(ocr_files, gt_files)

    preprocess_methods = PREPROCESSING_STEPS

    if experiment_type == "variation":
        preprocess_methods = [method for method in preprocess_methods]
    elif experiment_type == "multiple_rounds":
        preprocess_methods = ['filter_multiple_rounds', 'non_local_means_multiple_rounds']

    combinations = list(itertools.chain(*map(lambda x: itertools.combinations(preprocess_methods, x),
                                             range(0, len(preprocess_methods) + 1))))

    if experiment_type == "multiple_rounds":
        combinations = [combo for combo in combinations if len(combo) <= 2]

    all_metrics = []
    i = 0
    total_iterations = len(combinations) * len(ocr_files_dict)
    start_time = datetime.now()
    for combo in combinations:
        for filename, ocr_file in ocr_files_dict.items():
            preprocess_steps = []
            original_file = store_file(ocr_file, ocr_file.filename)
            image_path = original_file[:-5] + ".jpeg"
            if ocr_file.content_type == 'application/pdf':
                if not os.path.exists(image_path):  # Checking if the image already exists
                    images_from_pdf = convert_from_path(original_file)
                    images_from_pdf[0].save(image_path, 'JPEG')
            file_to_process = image_path
            # Apply selected preprocessing methods via pipeline and perform OCR
            ocr_text = await process_and_read_image(file_to_process, combo)

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
    report.generate_report()
    return FileResponse(report.filename, media_type='text/csv',
                        headers={"Content-Disposition": f"attachment;filename={report.filename}"})


@app.post('/process_results')
async def process_csv(request: Request):
    form = await request.form()
    csv_file = form.get("file")
    metric = form.get("metric")

    # Check if file or metric is not provided
    if csv_file is None:
        raise HTTPException(status_code=400, detail="No file provided")
    if metric is None:
        raise HTTPException(status_code=400, detail="No metric provided")

    filename = store_file(csv_file, csv_file.filename)
    report = TextMetricsReport()
    result_filename = report.analyze_experiment(filename, metric)

    # Return the file
    return FileResponse(result_filename, media_type='text/csv')


@app.post("/extract_features")
async def extract_features_endpoint(images: List[UploadFile] = File(...)):
    logger.info('Received request for extract_features Endpoint')
    if len(images) == 0:
        logger.error('No images provided')
        raise HTTPException(status_code=400, detail="No images provided")

    # DataFrame to hold the features for all images
    all_features = pd.DataFrame()

    for image in images:
        filename = store_file(image, image.filename)

        if image.content_type == 'application/pdf':
            images_from_pdf = convert_from_path(filename)
            image_path = filename[:-4] + ".jpeg"
            images_from_pdf[0].save(image_path, 'JPEG')
            filename = image_path

        # Extract features from the image
        extractor = ImageFeaturesExtractor(filename)
        features = extractor.extract_features()

        # Add the features to the DataFrame
        all_features = pd.concat([all_features, features], ignore_index=True)

        os.remove(filename)

    # Convert the DataFrame to a CSV
    csv = all_features.to_csv(index=False)
    file_bytes = BytesIO(csv.encode())
    file_name = 'features.csv'
    return Response(content=file_bytes.getvalue(), media_type="text/csv",
                    headers={"Content-Disposition": f"attachment;filename={file_name}"})

@app.post("/upload_and_train")
async def upload_and_train(file: UploadFile = File(...)):
    # Check if the file is in the correct format
    if file.content_type != 'text/csv':
        raise HTTPException(status_code=400, detail="File must be in CSV format")

    # Read the file into a DataFrame
    try:
        df = pd.read_csv(io.StringIO(file.file.read().decode("utf-8")))
        df.to_csv(file.filename, index=False)  # save the dataframe to a file to be used by the class
    except Exception as e:
        raise HTTPException(status_code=400, detail="Error reading CSV file") from e

    preprocessing_optimizer = PreprocessingOptimization(df)
    preprocessing_optimizer.load_and_preprocess_data()
    preprocessing_optimizer.train()

    return {"message": "Training completed. Check logs for model performance."}

async def process_and_read_image(image_path, preprocess_steps=None):
    if preprocess_steps is None:
        preprocess_steps = []
    preprocess_pipeline = ImagePipeline(image_path, preprocess_steps)
    preprocessed_image = preprocess_pipeline.process_image_variation()
    return await read_image(preprocessed_image)

