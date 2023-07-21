
import nltk
import logging
logger = logging.getLogger(__name__)

class TextMetrics:
    def __init__(self, ground_truth, ocr_text):
        self.ground_truth = ground_truth
        self.ocr_text = ocr_text

    def wer(self):
        return round(nltk.edit_distance(self.ground_truth.split(), self.ocr_text.split()) / len(self.ground_truth.split()), 2)

    def cer(self):
        return round(nltk.edit_distance(self.ground_truth, self.ocr_text) / len(self.ground_truth), 2)

    def lev_distance(self):
        return round(nltk.edit_distance(self.ground_truth, self.ocr_text), 2)
