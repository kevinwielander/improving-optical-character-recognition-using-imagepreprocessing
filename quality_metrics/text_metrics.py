import nltk


class TextMetrics:
    def __init__(self, ground_truth, ocr_text):
        self.ground_truth = ground_truth
        self.ocr_text = ocr_text

    def wer(self):
        return nltk.edit_distance(self.ground_truth.split(), self.ocr_text.split()) / len(self.ground_truth.split())

    def cer(self):
        return nltk.edit_distance(self.ground_truth, self.ocr_text) / len(self.ground_truth)

    def lev_distance(self):
        return nltk.edit_distance(self.ground_truth, self.ocr_text)
