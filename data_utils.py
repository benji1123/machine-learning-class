import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt
from typing import List, Tuple

from LinearRegressionModel import LRModel


def generate_data(bins: int, min_x=0, max_x=1, max_m=10) -> Tuple[ndarray, ndarray]:
    """Generate feature matrix (X), where each row is [x, x^2, ..., x^m]."""
    x_values = np.linspace(min_x, max_x, num=bins)  # base x-values
    training_data = []
    for x in x_values:
        # raise base-x values to higher powers to increase num features
        feature = [1] + [x ** i for i in range(1, max_m + 1)]
        training_data.append(feature)
    return np.array(training_data), generate_labels(bins)


def generate_labels(bins: int) -> ndarray:
    """Generate labels from hard-coded sinusoidal function."""
    return np.sin(4 * np.pi * np.linspace(0.0, 1.0, bins)) + 0.3 + np.random.randn(bins)


def plot_error_vs_m(
    models: List[LRModel], feature_matrix: ndarray, labels: ndarray, title: str) -> None:
    model_names = [str(model) for model in models]
    errors = []
    for m, model in enumerate(models):
        _feature_matrix = feature_matrix[:, 0:m+2]
        errors.append(model.compute_rss(_feature_matrix, labels))
    plt.plot(model_names, errors)
    plt.title(title)
    plt.show()
