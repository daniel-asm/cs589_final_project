"""
Original Author: Daniel Aziev-Smalovschi
"""
import numpy as np
from collections import Counter

class KNN:
    def __init__(self, k_neighbors=3, **kwargs):
        if k_neighbors <= 0:
            raise ValueError("The number of neighbors (k) must be positive.")
            
        self.k = k_neighbors
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train, **kwargs):
        if X_train.size == 0 or y_train.size == 0:
            raise ValueError("Training data cannot be empty.")

        self.X_train = X_train.T
        self.y_train = y_train.flatten()

    def predict(self, X_test):
        if self.X_train is None or self.y_train is None:
            raise RuntimeError("The model must be fitted with training data before making predictions.")
            
        X_test_T = X_test.T
        
        predictions = [self._predict_single(x) for x in X_test_T]
        
        return np.array(predictions)

    def _predict_single(self, x):
        distances = np.sqrt(np.sum((self.X_train - x)**2, axis=1))

        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = self.y_train[k_indices]
        
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]