import os

import nltk


class TextMetrics:
    def __init__(self, filename, ocr_text):
        self.filename = filename
        self.ocr_text = ocr_text
        self.ground_truth = self._read_ground_truth()

    def _read_ground_truth(self):
        base_path = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(base_path, 'resources', self.filename), 'r') as f:
            return f.read()

    def wer(self):
        return nltk.edit_distance(self.ground_truth.split(), self.ocr_text.split()) / len(self.ground_truth.split())

    def cer(self):
        return nltk.edit_distance(self.ground_truth, self.ocr_text) / len(self.ground_truth)

    def lev_distance(self):
        return nltk.edit_distance(self.ground_truth, self.ocr_text)
