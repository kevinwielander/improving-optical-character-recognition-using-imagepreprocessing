import numpy as np
import pandas as pd
import os
from datetime import datetime

import logging
from quality_metrics.text_metrics import TextMetrics
from quality_metrics.visualization import Visualization

logger = logging.getLogger(__name__)


class TextMetricsReport:
    def __init__(self, ground_truths=None, ocr_texts=None, filenames=None, preprocess_steps=None, all_metrics=None):
        logger.info('Initialized TextMetricsReport')
        self.ground_truths = ground_truths
        self.ocr_texts = ocr_texts
        self.filenames = filenames
        self.preprocess_steps = preprocess_steps
        self.all_metrics = all_metrics
        self.metrics = []
        self.filename = None

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

    def write_to_csv(self):
        current_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        self.filename = f'text_metrics_report.csv'
        directory = "resources/reports"
        if not os.path.exists(directory):
            os.makedirs(directory)

        self.filename = os.path.join(directory, f'text_metrics_report_{current_time}.csv')

        df = pd.DataFrame(self.all_metrics)

        if os.path.isfile(self.filename):
            df.to_csv(self.filename, mode='a', header=False, index=False)
            logger.info(f"Added metrics to existing report {self.filename}")
        else:
            logger.info(df.head())
            df.to_csv(self.filename, index=False)
            logger.info(f"Created new report {self.filename}")

    def analyze_experiment(self, filename, metric='Levenshtein Distance'):
        logger.info(f'Starting analysis of experiment with file: {filename} and metric: {metric}')
        # Convert the file to a DataFrame
        df = pd.read_csv(filename, sep=';')
        logger.info(f'Read data from {filename}')

        # Initialize a list to store the new data
        new_data = []
        improvements = []

        # Get all unique image numbers
        image_numbers = df['Filename'].unique()
        logger.info(f'Found {len(image_numbers)} unique image numbers')

        # For each image number
        for num in image_numbers:
            # Filter rows for current image number
            image_data = df[df['Filename'] == num]

            # Get the row with 'No preprocessing' (baseline)
            baseline_row = image_data[image_data['Preprocessing Steps'] == 'No preprocessing']

            # Get the baseline metric
            baseline_metric = baseline_row[metric].values[0]

            # Get the row with the best (minimum) metric
            best_row = image_data[image_data[metric] == image_data[metric].min()]

            # Get the best metric and corresponding preprocessing steps
            best_metric = best_row[metric].values[0]
            best_preprocessing = best_row['Preprocessing Steps'].values[0]

            # Calculate the improvement in percent and round to 2 decimal places
            improvement = round((baseline_metric - best_metric) / baseline_metric * 100, 2)

            # Append the improvement to the improvements list
            improvements.append(improvement)

            # Append the data for this image number to the list
            new_data.append([num, baseline_metric, best_metric, best_preprocessing, improvement])

        logger.info('Calculated improvements for all image numbers')

        # Calculate the average, median, min, max, and standard deviation of the improvements
        average_improvement = round(np.average(improvements), 2)
        median_improvement = round(np.median(improvements), 2)
        min_improvement = round(np.min(improvements), 2)
        max_improvement = round(np.max(improvements), 2)
        std_dev_improvement = round(np.std(improvements), 2)

        # Find the best and worst preprocessing steps
        best_preprocessing_steps = max(new_data, key=lambda x: x[-1])[3]
        worst_preprocessing_steps = min(new_data, key=lambda x: x[-1])[3]

        logger.info('Calculated statistics for improvements')

        # Append the calculated statistics to the list
        placeholder = '---'
        new_data.append([placeholder, placeholder, placeholder, 'Average Improvement:', average_improvement])
        new_data.append([placeholder, placeholder, placeholder, 'Median Improvement:', median_improvement])
        new_data.append([placeholder, placeholder, placeholder, 'Min Improvement:', min_improvement])
        new_data.append([placeholder, placeholder, placeholder, 'Max Improvement:', max_improvement])
        new_data.append([placeholder, placeholder, placeholder, 'Std Dev Improvement:', std_dev_improvement])
        new_data.append([placeholder, placeholder, placeholder, 'Best Preprocessing Steps:', best_preprocessing_steps])
        new_data.append(
            [placeholder, placeholder, placeholder, 'Worst Preprocessing Steps:', worst_preprocessing_steps])

        # Convert the list to a DataFrame
        new_df = pd.DataFrame(new_data, columns=['Image Number', 'Baseline ' + metric, 'Best ' + metric,
                                                 'Preprocessing Steps for Best ' + metric, 'Improvement in Percent'])

        vis = Visualization(new_df)
        vis.plot_histogram(save=False)

        directory = "resources/reports"
        if not os.path.exists(directory):
            os.makedirs(directory)
        self.filename = os.path.join(directory, f'text_metrics_report_output.csv')
        new_df.to_csv(self.filename, sep=';', index=False)

        logger.info(f'Finished analysis of experiment. Results saved to {self.filename}')

        return self.filename


