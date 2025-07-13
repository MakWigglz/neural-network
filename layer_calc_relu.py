import numpy as np
# Activation functions
def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Inputs
x = np.array([[1.0, 2.0]])  # shape: (1x2)
#  Weights and biases
W1 = np.array([[0.1, 0.4, 0.7],    # shape: (2x3)
               [0.2, 0.5, 0.8]])
b1 = np.array([[0.1, 0.2, 0.3]])   # shape: (1x3)
W2 = np.array([[-0.1],             # shape: (3x1)
               [ 0.3],
               [ 0.6]])
b2 = np.array([[-0.2]])            # shape: (1x1)

#  Forward Pass
# Input to hidden
z1 = np.dot(x, W1) + b1    # shape: (1x3)
a1 = relu(z1)              # shape: (1x3)

# Hidden to output
z2 = np.dot(a1, W2) + b2   # shape: (1x1)
output = sigmoid(z2)       # shape: (1x1)

#  Result
print("Network output (pass probability):", output[0][0])
