import numpy as np
import warnings
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score

def cross_validate(model_class, X, y, k=10, model_params=None, fit_params=None):
    """
    Universal cross-validation loop for any machine learning algorithm.
    """
    if model_params is None:
        model_params = {}
    if fit_params is None:
        fit_params = {}
        
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    
    accuracies = []
    f1_scores = []

    X_T = X.T 
    y_flat = y.flatten()
    
    for train_index, test_index in skf.split(X_T, y_flat):
        X_train, X_test = X_T[train_index].T, X_T[test_index].T
        y_train, y_test = y_flat[train_index].reshape(1, -1), y_flat[test_index].reshape(1, -1)

        model = model_class(**model_params)
        
        model.fit(X_train, y_train, **fit_params)
        
        predictions = model.predict(X_test)
        
        y_true_flat = y_test.flatten()
        y_pred_flat = predictions.flatten()
        
        accuracies.append(accuracy_score(y_true_flat, y_pred_flat))
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f1_scores.append(f1_score(y_true_flat, y_pred_flat, average='weighted'))
        
    return np.mean(accuracies), np.mean(f1_scores)