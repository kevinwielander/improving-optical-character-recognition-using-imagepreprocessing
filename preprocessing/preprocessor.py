import cv2
import numpy as np
from PIL import Image
import logging

logger = logging.getLogger(__name__)

class Preprocessor:
    def __init__(self, image_path):
        self.image_path = image_path
        self.image = cv2.imread(image_path)

    def to_grayscale(self):
        gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.image = gray_image
        logger.info("Image shape after grayscale conversion: %s", self.image.shape)

    def check_and_scale_dpi(self):
        with Image.open(self.image_path) as img:
            dpi = img.info.get('dpi')
        if dpi is None:
            dpi = (72, 72)
        current_dpi = max(dpi)
        if current_dpi < 300:
            scaling_factor = 300.0 / current_dpi
            resized_image = cv2.resize(self.image, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_CUBIC)
            self.image = resized_image

    def apply_filter(self):
        img = cv2.GaussianBlur(self.image, (5, 5), 0)
        self.image = img

    def apply_non_local_means(self):
        img = cv2.fastNlMeansDenoising(self.image)
        self.image = img

    def apply_morphological_operation(self):
        kernel = np.ones((1, 1), np.uint8)
        img = cv2.dilate(self.image, kernel, iterations=1)
        img = cv2.erode(img, kernel, iterations=1)
        self.image = img

    def apply_thresholding(self):
        img = cv2.threshold(self.image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        self.image = img


class ImagePipeline:
    def __init__(self, preprocessor):
        self.preprocessor = preprocessor

    def process_image(self):
        self.preprocessor.check_and_scale_dpi()
        self.preprocessor.to_grayscale()
        self.preprocessor.apply_filter()
        self.preprocessor.apply_non_local_means()
        self.preprocessor.apply_morphological_operation()
        self.preprocessor.apply_thresholding()
        processed_image = self.preprocessor.image
        return processed_image
