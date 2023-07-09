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
        # Convert the image to grayscale using OpenCV
        gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.image = gray_image
        logger.info("Image shape after grayscale conversion: %s", self.image.shape)

    def check_and_scale_dpi(self):
        # Open the image using PIL and get its DPI
        with Image.open(self.image_path) as img:
            dpi = img.info.get('dpi')

        # If the DPI is not available, assume it's 72 DPI
        if dpi is None:
            dpi = (72, 72)

        # Calculate the current DPI of the image
        current_dpi = max(dpi)

        # If the current DPI is less than 300, scale the image to 300 DPI
        if current_dpi < 300:
            # Calculate the scaling factor to convert the image to 300 DPI
            scaling_factor = 300.0 / current_dpi

            # Resize the image using OpenCV
            resized_image = cv2.resize(self.image, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_CUBIC)

            self.image = resized_image

    def apply_filter(self):
        # Apply blur to smooth out the edges
        img = cv2.GaussianBlur(self.image, (5, 5), 0)
        self.image = img

    def apply_morphological_operation(self):
        # Apply dilation and erosion to remove some noise
        kernel = np.ones((1, 1), np.uint8)
        img = cv2.dilate(self.image, kernel, iterations=1)
        img = cv2.erode(img, kernel, iterations=1)
        self.image = img

    def apply_thresholding(self):
        # Apply threshold to get image with only b & w (binarization)
        img = cv2.threshold(self.image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        self.image = img

class ImagePipeline:
    def __init__(self, preprocessor):
        self.preprocessor = preprocessor

    def process_image(self):
        # Call the preprocessing methods in order
        self.preprocessor.check_and_scale_dpi()
        self.preprocessor.to_grayscale()
        self.preprocessor.apply_filter()
        self.preprocessor.apply_morphological_operation()
        self.preprocessor.apply_thresholding()

        # Get the processed image from the Preprocessor object
        processed_image = self.preprocessor.image

        # Do additional processing steps as needed

        # Return the final processed image
        return processed_image
