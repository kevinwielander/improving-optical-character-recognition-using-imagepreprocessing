import re
import cv2
import numpy as np
import pandas as pd
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops

from skimage.filters import sobel
from skimage.color import rgb2gray
from scipy.stats import skew, kurtosis
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


class ImageFeaturesExtractor:
    def __init__(self, filename: str):
        logger.info('Iniialized Image FeatureExtractor')
        match = re.search(r'(\d{2}-\d{2})(?!.*\d)', filename)
        if match is not None:
            self.filename = match.group()
        else:
            self.filename = None
        self.img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

    def calculate_statistical_features(self):
        return np.mean(self.img), np.std(self.img), skew(np.array(self.img).flatten()), kurtosis(
            np.array(self.img).flatten())

    def calculate_contrast(self):
        min_val = np.min(self.img)
        max_val = np.max(self.img)
        contrast = max_val - min_val
        return contrast

    def calculate_noise(self):
        return np.std(self.img)

    def calculate_edge_density(self):
        edges = cv2.Canny(self.img, 100, 200)
        edge_density = np.sum(edges) / edges.size
        return edge_density

    def calculate_texture_features(self):
        lbp = local_binary_pattern(self.img, 8, 1, method='uniform')
        return np.histogram(lbp, bins=np.arange(0, 11), density=True)[0]

    def calculate_glcm_features(self):
        glcm = greycomatrix(self.img, [5], [0], 256, symmetric=True, normed=True)
        contrast = greycoprops(glcm, 'contrast')[0, 0]
        correlation = greycoprops(glcm, 'correlation')[0, 0]
        energy = greycoprops(glcm, 'energy')[0, 0]
        homogeneity = greycoprops(glcm, 'homogeneity')[0, 0]
        return contrast, correlation, energy, homogeneity

    def extract_features(self) -> pd.DataFrame:
        mean, std, skewness, kurtosis_val = self.calculate_statistical_features()
        contrast = self.calculate_contrast()
        noise = self.calculate_noise()
        edge_density = self.calculate_edge_density()
        texture_features = self.calculate_texture_features()
        image_dimensions = self.img.shape[:2]
        glcm_contrast, glcm_correlation, glcm_energy, glcm_homogeneity = self.calculate_glcm_features()

        df = pd.DataFrame({
            'filename': [self.filename],
            'mean': [mean],
            'std': [std],
            'skewness': [skewness],
            'kurtosis': [kurtosis_val],
            'contrast': [contrast],
            'noise': [noise],
            'edge_density': [edge_density],
            'texture_feature_1': [texture_features[0]],
            'texture_feature_2': [texture_features[1]],
            'texture_feature_3': [texture_features[2]],
            'glcm_contrast': [glcm_contrast],
            'glcm_correlation': [glcm_correlation],
            'glcm_energy': [glcm_energy],
            'glcm_homogeneity': [glcm_homogeneity],
        })

        return df
