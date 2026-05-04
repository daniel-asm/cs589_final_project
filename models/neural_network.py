""" 
Original Author: Daniel Aziev-Smalovschi
"""
import numpy as np

class NeuralNetwork:
    def __init__(self, layer_sizes, regularization_param=0.0):
        self.layer_sizes = layer_sizes
        self.lmbda = regularization_param
        self.num_layers = len(layer_sizes)
        self.weights = {}
        
        for l in range(1, self.num_layers):
            fan_in = self.layer_sizes[l-1]
            self.weights[l] = np.random.randn(self.layer_sizes[l], fan_in + 1) / np.sqrt(fan_in)

    def sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def sigmoid_derivative(self, a):
        return a * (1 - a)

    def compute_cost(self, y_true, a_out):
        m = y_true.shape[1]
        eps = 1e-15
        
        if y_true.shape[0] == 1:
            cost = -(1/m) * np.sum(y_true * np.log(a_out + eps) + (1 - y_true) * np.log(1 - a_out + eps))
        else:
            cost = -(1/m) * np.sum(y_true * np.log(a_out + eps))
        
        reg_term = 0
        for l in range(1, self.num_layers):
            reg_term += np.sum(np.square(self.weights[l][:, 1:]))
            
        cost += (self.lmbda / (2 * m)) * reg_term
        return cost

    def forward_propagation(self, X):
        activations = {0: X}
        
        A = X
        for l in range(1, self.num_layers):
            m = A.shape[1]
            A_bias = np.vstack([np.ones((1, m)), A])
            Z = np.dot(self.weights[l], A_bias)
            
            A = self.sigmoid(Z)
            activations[l] = A
            
        return activations

    def back_propagation(self, y, activations):
        m = y.shape[1]
        grads = {}
        deltas = {}
        
        A_out = activations[self.num_layers - 1]
        deltas[self.num_layers - 1] = A_out - y 
        
        for l in range(self.num_layers - 2, 0, -1):
            W_next = self.weights[l + 1]
            W_next_no_bias = W_next[:, 1:]
            delta_next = deltas[l + 1]
            A_current = activations[l]
            g_prime = self.sigmoid_derivative(A_current)
            deltas[l] = np.dot(W_next_no_bias.T, delta_next) * g_prime
        
        for l in range(1, self.num_layers):
            A_prev = activations[l-1]
            A_prev_bias = np.vstack([np.ones((1, m)), A_prev])
            dW = (1/m) * np.dot(deltas[l], A_prev_bias.T)
            dW[:, 1:] += (self.lmbda / m) * self.weights[l][:, 1:]
            grads[l] = dW
            
        return grads
        
    def predict(self, X):
        activations = self.forward_propagation(X)
        A_out = activations[self.num_layers - 1]
        
        if A_out.shape[0] == 1:
            return (A_out > 0.5).astype(int)
        else:
            return np.argmax(A_out, axis=0).reshape(1, -1)
    
    def update_weights(self, grads, alpha):
        for l in range(1, self.num_layers):
            self.weights[l] -= alpha * grads[l]

    def fit(self, X_train, y_train, alpha=0.1, epochs=1000, epsilon=1e-5, batch_size=32):
        m = X_train.shape[1]
        cost_history = []
        patience_counter = 0 
        
        for epoch in range(epochs):
            permutation = np.random.permutation(m)
            X_shuffled = X_train[:, permutation]
            y_shuffled = y_train[:, permutation]
            
            for i in range(0, m, batch_size):
                X_batch = X_shuffled[:, i:i+batch_size]
                y_batch = y_shuffled[:, i:i+batch_size]
                
                activations = self.forward_propagation(X_batch)
                grads = self.back_propagation(y_batch, activations)
                self.update_weights(grads, alpha)
            
            full_activations = self.forward_propagation(X_train)
            current_cost = self.compute_cost(y_train, full_activations[self.num_layers - 1])
            cost_history.append(current_cost)
            
            if epoch > 0:
                cost_diff = cost_history[-2] - cost_history[-1]
                if 0 < cost_diff < epsilon:
                    patience_counter += 1
                else:
                    patience_counter = 0
                    
                if patience_counter >= 3:
                    break
                
        return cost_history