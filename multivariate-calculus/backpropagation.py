import numpy as np
import matplotlib.pyplot as plt


SIGMA = lambda z: 1 / (1 + np.exp(-z))
D_SIGMA = lambda z: np.cosh(z/2) ** -2 / 4


def reset_network(n1 = 6, n2 = 7, random=np.random):
    """
    This function initialises the network with it's structure, it also resets any training already
    done.
    """
    global W1, W2, W3, b1, b2, b3
    W1 = random.randn(n1, 1) / 2
    W2 = random.randn(n2, n1) / 2
    W3 = random.randn(2, n2) / 2
    b1 = random.randn(n1, 1) / 2
    b2 = random.randn(n2, 1) / 2
    b3 = random.randn(2, 1) / 2

def network_function(a0):
    """
    This function feeds forward each activation to the next layer. It returns all weighted sums and activations.
    """
    z1 = W1 @ a0 + b1
    a1 = SIGMA(z1)
    z2 = W2 @ a1 + b2
    a2 = SIGMA(z2)
    z3 = W3 @ a2 + b3
    a3 = SIGMA(z3)
    return a0, z1, a1, z2, a2, z3, a3

def cost(x, y):
    """
    This is the cost function of a neural network with respect to a training set.
    """
    return np.linalg.norm(network_function(x)[-1] - y)**2 / x.size


def jacobian_w3(x, y):
    """
    Jacobian for third layer weights.

    Args:
        x: vector of inputs
        y: vector of desired outputs

    Returns:
        Jacobian for third layer weights
    """
    a0, z1, a1, z2, a2, z3, a3 = network_function(x)

    dCda3 = 2 * (a3 - y)
    da3dz3 = D_SIGMA(z3)
    dz3dW3 = a2

    dCdW3 = dCda3 * da3dz3 * dz3dW3
    J_W3 = dCdW3 @ a2.T / x.size

    return J_W3


def jacobian_w2(x, y):
    """
    Jacobian for second layer weights.

    Args:
        x: vector of inputs
        y: vector of desired outputs

    Returns:
        Jacobian for second layer weights
    """
    a0, z1, a1, z2, a2, z3, a3 = network_function(x)

    dCda2 = 2 * (a3 - y)
    da3da2 = D_SIGMA(z3) * W3
    da3dz3 = D_SIGMA(z3)
    dz3dW3 = a2

    dCdW2 = dCda3 * da3da2 * da2dz2 * dz2dW2
    J_W2 = dCdW2 @ a2.T / x.size

    return J_W3
