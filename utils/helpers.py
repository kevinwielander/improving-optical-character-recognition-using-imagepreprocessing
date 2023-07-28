import os
import re
from typing import List

from fastapi import UploadFile, HTTPException
from utils.config import TEMP_PATH
import logging

logger = logging.getLogger(__name__)


def prepare_file_dicts(ocr_files: List[UploadFile], gt_files: List[UploadFile]):
    # Use a regular expression to get the number at the end of the filename
    ocr_files_dict = {re.search(r'(\d{2}-\d{2})(?!.*\d)', file.filename).group(): file for file in ocr_files}
    gt_files_dict = {re.search(r'(\d{2}-\d{2})(?!.*\d)', file.filename).group(): file.file.read().decode('utf-8') for file in gt_files}

    return ocr_files_dict, gt_files_dict


def store_file(file, filename):
    filepath = os.path.join(TEMP_PATH, filename)

    # Only store the file if it doesn't exist
    if not os.path.exists(filepath):
        try:
            with open(filepath, "wb+") as file_object:
                file_object.write(file.file.read())
        except IOError as e:
            logger.error(f'Failed to store file {filename}: {str(e)}')
            raise HTTPException(status_code=500, detail="Internal server error")

    return filepath


