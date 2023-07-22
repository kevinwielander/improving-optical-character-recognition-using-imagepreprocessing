import os
from shutil import copyfile

import cv2
import numpy as np
from PIL import Image
import logging

logger = logging.getLogger(__name__)

class Preprocessor:
    def __init__(self, image_path):
        logger.info('Initialized Preprocessor')
        self.image_path = image_path
        self.copy_image_path = self._create_copy(image_path)
        self.image = cv2.imread(self.copy_image_path)

    def _create_copy(self, image_path):
        # generate a new path for the copy
        directory, filename = os.path.split(image_path)
        basename, ext = os.path.splitext(filename)
        new_filename = basename + '_copy' + ext
        new_path = os.path.join(directory, new_filename)

        # copy the image
        copyfile(image_path, new_path)

        return new_path

    def delete_copy(self):
        if os.path.exists(self.copy_image_path):
            os.remove(self.copy_image_path)
        self.image = None

    def to_grayscale(self):
        gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.image = gray_image
        logger.info("Image converted to grayscale. Shape: %s", self.image.shape)

    def check_and_scale_dpi(self):
        with Image.open(self.copy_image_path) as img:
            img.load()  # Load image data into memory
            dpi = img.info.get('dpi')
        if dpi is None:
            dpi = (72, 72)
        current_dpi = max(dpi)
        if current_dpi < 300:
            scaling_factor = 300.0 / current_dpi
            img = img.resize((int(img.size[0] * scaling_factor), int(img.size[1] * scaling_factor)), Image.ANTIALIAS)
            logger.info("Checked and scaled DPI. Initial DPI: %s", current_dpi)

        self.image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    def apply_filter(self):
        img = cv2.GaussianBlur(self.image, (5, 5), 0)
        self.image = img
        logger.info("Applied Gaussian filter.")

    def apply_non_local_means(self):
        img = cv2.fastNlMeansDenoising(self.image)
        self.image = img
        logger.info("Applied Non-Local Means Denoising.")

    def apply_morphological_operation(self):
        kernel = np.ones((1, 1), np.uint8)
        img = cv2.dilate(self.image, kernel, iterations=1)
        img = cv2.erode(img, kernel, iterations=1)
        self.image = img
        logger.info("Applied morphological operations.")

    def apply_thresholding(self):
        img = cv2.threshold(self.image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        self.image = img
        logger.info("Applied thresholding.")
