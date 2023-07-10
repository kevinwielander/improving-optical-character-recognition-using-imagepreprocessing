import pandas as pd
import os
from datetime import datetime

import logging
from quality_metrics.text_metrics import TextMetrics
from quality_metrics.visualization import Visualization

logger = logging.getLogger(__name__)

class TextMetricsReport:
    def __init__(self, ground_truths, ocr_texts, filenames, preprocess_steps):
        self.ground_truths = ground_truths
        self.ocr_texts = ocr_texts
        self.filenames = filenames
        self.preprocess_steps = preprocess_steps
        self.filename = None
        self.metrics = []

    def generate_report(self):
        logger.info('Generating text metrics report')
        current_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        self.filename = f'resources/text_metrics_report_{current_time}.csv'

        total_cer, total_wer, total_lev_distance = 0, 0, 0
        total_characters, total_words = 0, 0
        for i, (gt, ocr, fname, steps) in enumerate(
                zip(self.ground_truths, self.ocr_texts, self.filenames, self.preprocess_steps)):
            tm = TextMetrics(gt, ocr)
            wer = tm.wer()
            cer = tm.cer()
            lev_distance = tm.lev_distance()

            total_wer += wer * len(ocr.split())
            total_cer += cer * len(ocr.replace(' ', ''))
            total_lev_distance += lev_distance
            total_characters += len(ocr.replace(' ', ''))
            total_words += len(ocr.split())

            self.metrics.append(
                {'Index': i, 'Filename': fname, 'Preprocessing Steps': ', '.join(steps), 'WER': wer, 'CER': cer,
                 'Levenshtein Distance': lev_distance})

        logger.info(f"Computed metrics for {len(self.metrics)} files")

        if total_words != 0 and total_characters != 0:
            overall_wer = total_wer / total_words
            overall_cer = total_cer / total_characters
            overall_lev_distance = total_lev_distance / len(self.metrics)
            self.metrics.append({'Index': -1, 'Filename': 'Overall', 'WER': overall_wer, 'CER': overall_cer,
                                 'Levenshtein Distance': overall_lev_distance})

        df = pd.DataFrame(self.metrics)

        # Create the directory if it does not exist
        directory = "resources/reports"
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Create the filename with the directory
        self.filename = os.path.join(directory, f'text_metrics_report_{current_time}.csv')

        if os.path.isfile(self.filename):
            df.to_csv(self.filename, mode='a', header=False, index=False)
            logger.info(f"Added metrics to existing report {self.filename}")
        else:
            df.to_csv(self.filename, index=False)
            logger.info(f"Created new report {self.filename}")

        vis = Visualization(df)
        vis.plot_metrics(save=False)
