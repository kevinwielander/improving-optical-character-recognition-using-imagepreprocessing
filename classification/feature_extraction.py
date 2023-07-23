import re

import cv2
import numpy as np
import pandas as pd
from skimage.feature import local_binary_pattern
from skimage.filters import sobel
from skimage.color import rgb2gray
from typing import Tuple
import logging

logger = logging.getLogger(__name__)

class ImageFeaturesExtractor:
    def __init__(self, filename: str):
        logger.info('Iniialized Image FeatureExtractor')
        self.filename = re.findall(r'(\d+)\.', filename)[-1]
        self.img = cv2.imread(filename)

    def calculate_contrast(self):
        min_val = np.min(self.img)
        max_val = np.max(self.img)
        contrast = max_val - min_val
        return contrast

    def calculate_noise(self):
        return np.std(self.img)

    def calculate_edge_density(self):
        gray_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray_img, 100, 200)
        edge_density = np.sum(edges) / edges.size
        return edge_density

    def calculate_color_diversity(self):
        return len(np.unique(self.img))

    def calculate_texture_features(self):
        gray_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        lbp = local_binary_pattern(gray_img, 8, 1, method='uniform')
        return np.histogram(lbp, bins=np.arange(0, 11), density=True)[0]

    def extract_features(self) -> pd.DataFrame:
        contrast = self.calculate_contrast()
        noise = self.calculate_noise()
        edge_density = self.calculate_edge_density()
        color_diversity = self.calculate_color_diversity()
        texture_features = self.calculate_texture_features()
        image_dimensions = self.img.shape[:2]

        df = pd.DataFrame({
            'filename': [self.filename],
            'contrast': [contrast],
            'noise': [noise],
            'edge_density': [edge_density],
            'color_diversity': [color_diversity],
            'image_width': [image_dimensions[1]],
            'image_height': [image_dimensions[0]],
            'texture_feature_1': [texture_features[0]],
            'texture_feature_2': [texture_features[1]],
            'texture_feature_3': [texture_features[2]],
        })

        return df
