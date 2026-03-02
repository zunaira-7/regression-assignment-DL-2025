import numpy as np

class LinearRegressionNetwork:
    def __init__(self, input_dim, hidden_dim):
        
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2 / input_dim)
        self.b1 = np.zeros((1, hidden_dim))

        self.W2 = np.random.randn(hidden_dim, 1) * np.sqrt(2 / hidden_dim)
        self.b2 = np.zeros((1, 1))

    def forward(self, X):
    
        Z1 = X @ self.W1 + self.b1
        Z2 = Z1 @ self.W2 + self.b2
        return Z2
