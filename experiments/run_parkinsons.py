import os
import numpy as np
import matplotlib.pyplot as plt
from utils.data_loader import load_tabular_dataset
from utils.evaluation import grid_search_cv
from models.neural_network import NeuralNetwork
from models.knn import KNN
from models.random_forest import RandomForest

def run_parkinsons_experiments():
    print("Loading Parkinson's Dataset...")

    X, y = load_tabular_dataset('datasets/parkinsons.csv', target_column='Diagnosis')

    os.makedirs('latex_source', exist_ok=True)

    # RANDOM FOREST ALGORITHM
    print("\nRunning Random Forest")
    rf_param_grid = {
        'ntree': [1, 5, 10, 20, 30, 40, 50],
    }

    rf_best_params, rf_results = grid_search_cv(RandomForest, X, y, rf_param_grid, k=10)

    print("\nRandom Forest Full Results")
    for res in rf_results:
        print(f"Params: {res['params']} | Accuracy: {res['accuracy']:.4f} | F1-Score: {res['f1']:.4f}")

    ntree_vals = []
    rf_f1_vals = []

    # Collect results for plotting
    for res in rf_results:
        ntree_vals.append(res['params']['ntree'])
        rf_f1_vals.append(res['f1'])

    plt.figure(figsize=(8, 6))
    plt.plot(ntree_vals, rf_f1_vals, marker='o', linestyle='-', linewidth=2, markersize=8)
    plt.title(f"Random Forest F1-Score vs. Number of Trees (Parkinson's)\nBest Params: {rf_best_params}")
    plt.xlabel("Number of Trees")
    plt.ylabel("F1-Score")
    plt.grid(True)
    plt.xticks(ntree_vals)
    plt.savefig('latex_source/rf_parkinsons_f1_curve.png')
    plt.show()

    print("\nRunning Neural Network")
    nn_param_grid = {
        'layer_sizes': [[22, 8, 1], [22, 16, 1], [22, 8, 4, 1]], 
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
    
    m = X.shape[1] 
    instances_seen = [epoch * m for epoch in range(1, len(cost_history) + 1)]
    
    plt.figure(figsize=(8, 6))
    plt.plot(instances_seen, cost_history, color='blue', linewidth=2)
    plt.title(f"Neural Network Convergence (Parkinson's)\nParams: {nn_best_params}")
    plt.xlabel("Number of Training Instances Presented")
    plt.ylabel("Cost (J)")
    plt.grid(True)
    plt.savefig('latex_source/nn_parkinsons_learning_curve.png')
    plt.show()

    print("\nRunning K-Nearest Neighbors")
    knn_param_grid = {
        'k_neighbors': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29]
    }
    
    knn_best_params, knn_results = grid_search_cv(KNN, X, y, knn_param_grid, k=10)

    print("\nKNN Full Results")
    for res in knn_results:
        print(f"k: {res['params']['k_neighbors']} | Accuracy: {res['accuracy']:.4f} | F1-Score: {res['f1']:.4f}")
    
    k_vals = [res['params']['k_neighbors'] for res in knn_results]
    f1_vals = [res['f1'] for res in knn_results]
    
    plt.figure(figsize=(8, 6))
    plt.plot(k_vals, f1_vals, marker='o', color='red', linestyle='dashed', linewidth=2, markersize=8)
    plt.title(f"KNN F1-Score vs. k (Parkinson's)\nBest k: {knn_best_params['k_neighbors']}")
    plt.xlabel("Number of Neighbors (k)")
    plt.ylabel("F1-Score")
    plt.grid(True)
    plt.xticks(k_vals)
    plt.savefig('latex_source/knn_parkinsons_f1_curve.png')
    plt.show()

if __name__ == "__main__":
    run_parkinsons_experiments()