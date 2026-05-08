import os
import numpy as np
import matplotlib.pyplot as plt
from utils.data_loader import load_tabular_dataset
from utils.evaluation import grid_search_cv

from models.random_forest import RandomForest
from models.decision_tree import DecisionTreeClassifier
from models.neural_network import NeuralNetwork

def run_credit_experiments():
    print("Loading Credit Approval...")

    X, y = load_tabular_dataset('datasets/credit_approval.csv', target_column='label')

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
    plt.title(f"Random Forest F1-Score vs. Number of Trees (Credit)\nBest Params: {rf_best_params}")
    plt.xlabel("Number of Trees")
    plt.ylabel("F1-Score")
    plt.grid(True)
    plt.xticks(ntree_vals)
    plt.savefig('latex_source/rf_credit_f1_curve.png')
    plt.show()

    # NEURAL NETWORK
    print("\nRunning Neural Network")
    input_size = X.shape[0]

    nn_param_grid = {
        'layer_sizes': [
            [input_size, 16, 8, 1],
            [input_size, 16, 16, 1],
            [input_size, 16, 8, 4, 1]
        ],
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
    plt.title(f"Neural Network Convergence (Credit Approval)\nParams: {nn_best_params}")
    plt.xlabel("Number of Training Instances Presented")
    plt.ylabel("Cost (J)")
    plt.grid(True)
    plt.savefig('latex_source/nn_credit_learning_curve.png')
    plt.show()

    # DECISION TREE
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

    plt.figure(figsize=(8, 6))
    plt.plot(depth_vals, dt_f1_vals, marker='s', color='orange', linestyle='-', linewidth=2, markersize=8)
    plt.title(f"Decision Tree F1-Score vs. Max Depth (Credit Approval)\nBest Params: {dt_best_params}")
    plt.xlabel("Max Depth (20 = Unbounded)")
    plt.ylabel("F1-Score")
    plt.grid(True)
    plt.xticks(depth_vals, labels=[str(d) if d != 20 else 'None' for d in depth_vals])
    plt.savefig('latex_source/dt_credit_f1_curve.png')
    plt.show()

if __name__ == "__main__":
    run_credit_experiments()