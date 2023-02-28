import pandas as pd
import os
from datetime import datetime

from quality_metrics.text_metrics import TextMetrics


class TextMetricsReport:
    def __init__(self, ground_truths, ocr_texts):
        self.ground_truths = ground_truths
        self.ocr_texts = ocr_texts
        self.filename = None
        self.metrics = []

    def generate_report(self):
        current_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        self.filename = f'resources/text_metrics_report_{current_time}.csv'

        for i, (gt, ocr) in enumerate(zip(self.ground_truths, self.ocr_texts)):
            tm = TextMetrics(gt, ocr)
            wer = tm.wer()
            cer = tm.cer()
            lev_distance = tm.lev_distance()
            self.metrics.append({'Index': i, 'Ground Truth': gt, 'OCR Text': ocr, 'WER': wer, 'CER': cer,
                                 'Levenshtein Distance': lev_distance})

        df = pd.DataFrame(self.metrics)
        if os.path.isfile(self.filename):
            df.to_csv(self.filename, mode='a', header=False, index=False)
        else:
            df.to_csv(self.filename, index=False)
