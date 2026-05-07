import os
import numpy as np
import matplotlib.pyplot as plt
from utils.data_loader import load_fashion_mnist_sample
from utils.evaluation import grid_search_cv
from models.neural_network import NeuralNetwork
from models.knn import KNN

def run_fashion_experiments():
    X, y = load_fashion_mnist_sample(sample_size=2000)
    
    os.makedirs('latex_source', exist_ok=True)

    # 1. Neural Network Experiment
    print("\nRunning Neural Network (Fashion-MNIST)")
    nn_param_grid = {
        'layer_sizes': [[784, 128, 10], [784, 64, 10]],
        'regularization_param': [0.0, 0.1, 0.5]
    }
    
    fit_params = {
        'alpha': 0.1,
        'epochs': 200,
        'batch_size': 128
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
    
    plt.figure(figsize=(8, 6))
    plt.plot(instances_presented, cost_history, color='purple', linewidth=2)
    plt.title(f"Neural Network Convergence (Fashion-MNIST)\nParams: {nn_best_params}")
    plt.xlabel("Number of Training Instances Presented")
    plt.ylabel("Cost (J)")
    plt.grid(True)
    plt.savefig('latex_source/nn_fashion_learning_curve.png')
    plt.show()


    # 2. kNN Experiment
    print("\nRunning K-Nearest Neighbors (Fashion-MNIST)")
    knn_param_grid = {
        'k_neighbors': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51]
    }
    
    knn_best_params, knn_results = grid_search_cv(KNN, X, y, knn_param_grid, k=3)
    
    print("\nKNN Full Results")
    for res in knn_results:
        print(f"k: {res['params']['k_neighbors']} | Accuracy: {res['accuracy']:.4f} | F1-Score: {res['f1']:.4f}")
    
    k_vals = [res['params']['k_neighbors'] for res in knn_results]
    f1_vals = [res['f1'] for res in knn_results]
    
    plt.figure(figsize=(8, 6))
    plt.plot(k_vals, f1_vals, marker='o', color='green', linestyle='dashed', linewidth=2, markersize=8)
    plt.title(f"KNN F1-Score vs. k (Fashion-MNIST)\nBest k: {knn_best_params['k_neighbors']}")
    plt.xlabel("Number of Neighbors (k)")
    plt.ylabel("F1-Score")
    plt.grid(True)
    plt.xticks(k_vals)
    plt.savefig('latex_source/knn_fashion_f1_curve.png')
    plt.show()

if __name__ == "__main__":
    run_fashion_experiments()