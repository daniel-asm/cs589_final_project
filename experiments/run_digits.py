import os
import numpy as np
import matplotlib.pyplot as plt
from utils.data_loader import load_digits_dataset
from utils.evaluation import grid_search_cv
from models.neural_network import NeuralNetwork
from models.knn import KNN
from models.decision_tree import DecisionTreeClassifier

def run_digits_experiments():
    print("Loading Digits Dataset...")
    X, y = load_digits_dataset()
    
    os.makedirs('latex_source', exist_ok=True)
    # 1. Neural Network Experiment
    print("\nRunning Neural Network")
    nn_param_grid = {
        'layer_sizes': [[64, 32, 10], [64, 16, 10]],
        'regularization_param': [0.0, 0.1, 0.5]
    }
    
    fit_params = {
        'alpha': 0.1,
        'epochs': 500,
        'batch_size': 32
    }
    
    nn_best_params, nn_results = grid_search_cv(NeuralNetwork, X, y, nn_param_grid, k=10, fit_params=fit_params)

    print("\nNeural Network Full Results")
    for res in nn_results:
        print(f"Params: {res['params']} | Accuracy: {res['accuracy']:.4f} | F1-Score: {res['f1']:.4f}")
    
    print("\nRetraining optimal NN to capture learning curve...")
    optimal_nn = NeuralNetwork(**nn_best_params)
    cost_history = optimal_nn.fit(X, y, **fit_params)

    m_samples = X.shape[1]
    instances_presented = [epoch * m_samples for epoch in range(1, len(cost_history) + 1)]
    
    # Plot NN Convergence
    plt.figure(figsize=(8, 6))
    plt.plot(instances_presented, cost_history, color='blue', linewidth=2)
    plt.title(f"Neural Network Convergence (Digits)\nParams: {nn_best_params}")
    plt.xlabel("Number of Training Instances")
    plt.ylabel("Cost (J)")
    plt.grid(True)
    plt.savefig('latex_source/nn_digits_learning_curve.png')
    plt.show()

    # 2. kNN Experiment
    print("\nRunning K-Nearest Neighbors")
    knn_param_grid = {
        'k_neighbors': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51]
    }
    
    knn_best_params, knn_results = grid_search_cv(KNN, X, y, knn_param_grid, k=10)

    print("\nKNN Full Results")
    for res in knn_results:
        print(f"k: {res['params']['k_neighbors']} | Accuracy: {res['accuracy']:.4f} | F1-Score: {res['f1']:.4f}")
    
    # Extract data for the KNN hyperparameter curve
    k_vals = [res['params']['k_neighbors'] for res in knn_results]
    f1_vals = [res['f1'] for res in knn_results]
    
    # Plot KNN F1 vs k
    plt.figure(figsize=(8, 6))
    plt.plot(k_vals, f1_vals, marker='o', color='red', linestyle='dashed', linewidth=2, markersize=8)
    plt.title(f"KNN F1-Score vs. k (Digits)\nBest k: {knn_best_params['k_neighbors']}")
    plt.xlabel("Number of Neighbors (k)")
    plt.ylabel("F1-Score")
    plt.grid(True)
    plt.xticks(k_vals)
    plt.savefig('latex_source/knn_digits_f1_curve.png')
    plt.show()

    # 3. Decision Tree Experiment
    print("\nRunning Decision Tree")
    dt_param_grid = {
        'max_depth': [3, 5, 7, 9, 11, 15, None],
        'min_samples_split': [2],
        'criterion': ['entropy', 'gini']
    }
    
    dt_best_params, dt_results = grid_search_cv(DecisionTreeClassifier, X, y, dt_param_grid, k=10)

    print("\nDecision Tree Full Results")
    for res in dt_results:
        print(f"Params: {res['params']} | Accuracy: {res['accuracy']:.4f} | F1-Score: {res['f1']:.4f}")
    
    best_crit = dt_best_params['criterion']
    depth_vals = []
    dt_f1_vals = []
    
    for res in dt_results:
        if res['params']['criterion'] == best_crit:
            val = res['params']['max_depth']
            plot_val = 20 if val is None else val 
            depth_vals.append(plot_val)
            dt_f1_vals.append(res['f1'])
            
    # Plot DT F1 vs Max Depth
    plt.figure(figsize=(8, 6))
    plt.plot(depth_vals, dt_f1_vals, marker='s', color='orange', linestyle='-', linewidth=2, markersize=8)
    plt.title(f"Decision Tree F1-Score vs. Max Depth (Digits)\nBest Params: {dt_best_params}")
    plt.xlabel("Max Depth (20 = Unbounded)")
    plt.ylabel("F1-Score")
    plt.grid(True)
    plt.xticks(depth_vals, labels=[str(d) if d != 20 else 'None' for d in depth_vals])
    plt.savefig('latex_source/dt_digits_f1_curve.png')
    plt.show()

if __name__ == "__main__":
    run_digits_experiments()