import nltk
import logging

logger = logging.getLogger(__name__)


class TextMetrics:
    def __init__(self, ground_truth, ocr_text):
        self.ground_truth = ground_truth
        self.ocr_text = ocr_text

    def wer(self):
        try:
            ground_truth_words = self.ground_truth.split()
            if len(ground_truth_words) == 0:
                logger.error('Division by zero error. Ground truth is empty.')
                return None
            else:
                wer_value = round(
                    nltk.edit_distance(ground_truth_words, self.ocr_text.split()) / len(ground_truth_words), 2)
                logger.info('Computed WER')
                return wer_value
        except Exception as e:
            logger.error(f'An error occurred: {str(e)}')
            return None

    def cer(self):
        try:
            if len(self.ground_truth) == 0:
                logger.error('Division by zero error. Ground truth is empty.')
                return None
            else:
                cer_value = round(nltk.edit_distance(self.ground_truth, self.ocr_text) / len(self.ground_truth), 2)
                logger.info('Computed CER')
                return cer_value
        except Exception as e:
            logger.error(f'An error occurred: {str(e)}')
            return None

    def lev_distance(self):
        try:
            lev_distance_value = round(nltk.edit_distance(self.ground_truth, self.ocr_text), 2)
            logger.info('Computed Levenshtein Distance')
            return lev_distance_value
        except Exception as e:
            logger.error(f'An error occurred: {str(e)}')
            return None
