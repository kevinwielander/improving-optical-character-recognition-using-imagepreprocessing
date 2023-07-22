import os

REPORTS_PATH = os.path.join("resources", "reports")
PLOTS_PATH = os.path.join("resources", "plots")
TEMP_PATH = os.path.join("resources", "temp")
LOGS_PATH = "logs"

LOG_FILE = "app.log"


# List of preprocessing steps
PREPROCESSING_STEPS = ['filter', 'non_local_means', 'morphological_operation', 'thresholding']
#PREPROCESSING_STEPS = ['filter']

PREPROCESSING_METHODS = {
    'filter': 'apply_filter',
    'non_local_means': 'apply_non_local_means',
    'morphological_operation': 'apply_morphological_operation',
    'thresholding': 'apply_thresholding'
}