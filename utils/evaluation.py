import numpy as np
import warnings
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
import itertools

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

def grid_search_cv(model_class, X, y, param_grid, k=10, fit_params=None):
    """
    Executes cross-validation over a grid of hyperparameters.
    param_grid: dict mapping parameter names to lists of values to test.
    """
    keys = param_grid.keys()
    values = param_grid.values()
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    best_f1 = -1
    best_params = None
    results = []
    
    for params in combinations:
        print(f"Testing {model_class.__name__} with params: {params}")
        acc, f1 = cross_validate(model_class, X, y, k=k, model_params=params, fit_params=fit_params)
        results.append({'params': params, 'accuracy': acc, 'f1': f1})
        
        if f1 > best_f1:
            best_f1 = f1
            best_params = params
            
    print(f"\nBest Parameters: {best_params} (F1: {best_f1:.4f})")
    return best_params, results