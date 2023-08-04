import cv2
import numpy as np
from PIL import Image
import logging
import os
import shutil
import pytesseract
from pytesseract import Output

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageProcessor:
    def __init__(self, image_path):
        self.image_path = image_path

    def to_grayscale(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def check_and_scale_dpi(self, img_path):
        with Image.open(img_path) as img:
            img.load()  # Load image data into memory
            dpi = img.info.get('dpi')
        if dpi is None:
            dpi = (72, 72)
        current_dpi = max(dpi)
        if current_dpi < 300:
            scaling_factor = 300.0 / current_dpi
            img = img.resize((int(img.size[0] * scaling_factor), int(img.size[1] * scaling_factor)), Image.LANCZOS)
            logger.info("Checked and scaled DPI. Initial DPI: %s", current_dpi)
            img.save(img_path)  # Save the scaled image back to the file

        return cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Reload the image from the file as grayscale

    def apply_non_local_means(self, img):
        return cv2.fastNlMeansDenoising(img)

    def apply_thresholding(self, img):
        return cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    def save_image(self, img, suffix):
        # Splits the filename and extension to insert the suffix before the extension
        filename, extension = os.path.splitext(self.image_path)
        new_image_path = f"{filename}_{suffix}{extension}"
        cv2.imwrite(new_image_path, img)
        return new_image_path

    def visualize_recognition(self, img):
        # Check if the image is grayscale. If it is, convert it to BGR.
        if len(img.shape) == 2 or img.shape[2] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        d = pytesseract.image_to_data(img, output_type=Output.DICT, lang='deu')
        n_boxes = len(d['level'])
        for i in range(n_boxes):
            if int(d['conf'][i]) > -1:  # Only consider recognized text with a confidence level greater than -1
                (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)  # Changed last parameter from 2 to 3

        # Save the image with drawn boxes to a file
        recognized_image_path = self.save_image(img, 'recognized')
        return img


if __name__ == "__main__":
    pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

    img_path = "../resources/dataset/Images_low_quality/image13-04.jpeg"

    processor = ImageProcessor(img_path)
    img = cv2.imread(img_path)

    # Save the initial image with boxes
    img_initial = img.copy()
    img_initial_boxes = processor.visualize_recognition(img_initial)
    processor.save_image(img_initial_boxes, 'initial')

    # Apply all preprocessing steps and save the final image with boxes
    img_gray = processor.to_grayscale(img.copy())
    img_gray_path = processor.save_image(img_gray, 'grayscale')

    img_scaled = processor.check_and_scale_dpi(img_gray_path)
    img_scaled_path = processor.save_image(img_scaled, 'scaled_dpi')

    img_nlmeans = processor.apply_non_local_means(img_scaled.copy())
    img_nlmeans_path = processor.save_image(img_nlmeans, 'non_local_means')

    img_threshold = processor.apply_thresholding(img_nlmeans.copy())
    img_threshold_path = processor.save_image(img_threshold, 'thresholded')

    img_recognized = processor.visualize_recognition(img_threshold.copy())
    processor.save_image(img_recognized, 'recognized')

    # If the images are not the same size, resize the smaller one to the size of the larger one
    max_height = max(img_initial_boxes.shape[0], img_recognized.shape[0])
    max_width = max(img_initial_boxes.shape[1], img_recognized.shape[1])

    img_initial_boxes_resized = cv2.resize(img_initial_boxes, (max_width, max_height))
    img_recognized_resized = cv2.resize(img_recognized, (max_width, max_height))

    # Stack the images side by side
    comparison_image = np.hstack((img_initial_boxes_resized, img_recognized_resized))

    # Save the side by side comparison image
    processor.save_image(comparison_image, 'comparison')

