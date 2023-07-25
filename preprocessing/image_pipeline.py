import logging

from preprocessing.preprocessor import Preprocessor
from utils.config import PREPROCESSING_METHODS

logger = logging.getLogger(__name__)


class ImagePipeline:
    def __init__(self, image_path, preprocess_steps):
        logger.info('Initialized Image Pipeline')
        self.preprocessor = Preprocessor(image_path)
        self.preprocess_steps = preprocess_steps

    def process_image(self):
        # Always apply grayscale and scaling
        self.preprocessor.check_and_scale_dpi()
        self.preprocessor.to_grayscale()

        method_map = {step: getattr(self.preprocessor, PREPROCESSING_METHODS[step]) for step in self.preprocess_steps}
        for step in self.preprocess_steps:
            method_map[step]()
        processed_image = self.preprocessor.image
        logger.info("Image preprocessing of Image Pipeline complete.")
        self.preprocessor.delete_copy()
        return processed_image

    def process_image_variation(self):
        self.preprocessor.check_and_scale_dpi()
        self.preprocessor.to_grayscale()

        method_map = {step: getattr(self.preprocessor, PREPROCESSING_METHODS.get(step)) for step in
                      self.preprocess_steps}
        for step in self.preprocess_steps:
            if step in method_map:
                method_map[step]()
        processed_image = self.preprocessor.image
        logger.info("Image preprocessing with variations complete.")
        self.preprocessor.delete_copy()
        return processed_image

    def process_image_multiple_rounds(self):
        self.preprocessor.check_and_scale_dpi()
        self.preprocessor.to_grayscale()

        method_map = {step: getattr(self.preprocessor, PREPROCESSING_METHODS[step]) for step in self.preprocess_steps}
        for step in self.preprocess_steps:
            if step in method_map:
                method_map[step]()
        processed_image = self.preprocessor.image
        logger.info("Image preprocessing with multiple rounds complete.")
        self.preprocessor.delete_copy()
        return processed_image

