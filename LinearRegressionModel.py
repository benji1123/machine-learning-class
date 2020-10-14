import numpy as np
from numpy import ndarray
import math
from typing import List


class LRModel:
    def __str__(self):
        return f"m={self.num_features}"

    def __init__(self, num_features, alpha, training_data, validation_data, reg=0):
        self.num_features = num_features
        self.alpha = alpha
        self.training_x, self.training_t = training_data
        self.validation_x, self.validation_t = validation_data
        self.weights = np.random.rand(self.num_features+1)
        self.reg = reg

    def compute_rss(self, feature_matrix, labels) -> float:
        """Implements cost function as residual sum of squares (RSS)."""
        prediction = np.matmul(feature_matrix, self.weights)
        squared_error = (prediction - labels)**2
        return np.average(squared_error)

    def update_weights(self) -> ndarray:
        """batch gradient descent update-step."""
        prediction = self.get_prediction(self.training_x)
        self.weights -= (2*self.alpha) * (
                np.average(np.matmul(prediction - self.training_t, self.training_x))
                + np.average(self.reg * self.weights))

    def get_prediction(self, feature_matrix) -> ndarray:
        """Compute prediction from feature-matrix and weights."""
        return np.matmul(feature_matrix, self.weights)
