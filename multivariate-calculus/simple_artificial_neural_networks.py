import numpy as np


def func_a1(a0, w1, b1, sigma=np.tanh):
    """
    Feed-forward equation, two node neural network.

    Args:
        a0: activation of input neuron
        sigma: activation function
        w1: weight of connection between neuron (0) and neuron (1)
        b1: bias of neuron (1)

    Returns:
        a1: activation of output neuron
    """
    z = w1 * a0 + b1
    return sigma(z)


def single_layer_neural_network_no_hidden_layer(a0, W, b, sigma=np.tanh):
    """
    Feed-forward equation, neural network with an input layer and an output layer.

    Args:
        a0: vector of input neurons
        sigma: activation function
        W: matrix of weights of connections between neurons in the input layer and the output layer
        b: vector of biases of output neurons

    Returns:
        a1: vector of output neurons
    """
    z = W @ a0 + b
    return sigma(z)


def single_layer_neural_network_with_hidden_layer(a0, W1, W2, b1, b2, sigma=np.tanh):
    """
    Feed-wordward equation, neural network with an input layer, a hidden layer, and an output layer

    Args:
        a0: vector of input neurons
        sigma: activation function
        W1: matrix of weights of connections between neurons in the input layer and the hidden layer
        W2: matrix of weights of connections between neurons in the input layer and the output layer
        b1: vector of biases of hidden layer neurons
        b2: vector of biases of output neurons

    Returns:
        a2: vector of output neurons
    """
    a1 = sigma(W1 @ a0 + b1)
    a2 = sigma(W2 @ a1 + b2)
    return a2


def cost_two_node_neural_network(a1, y):
    """
    Cost of training a two node neural network for a specific training example.

    Args:
        a1: activation of output neuron
        y: desired output

    Returns:
        cost: training cost
    """
    return (a1 - y) ** 2


def dCdw_two_node_neural_network(x, w1, b1, y):
    """
    Cost gradient with regard to weight for a two node neural network.

    Args:
        x: input
        y: desired output
        w1: weight of connection between neuron (0) and neuron (1)
        b1: bias of neuron (1)

    Returns:
        dCdw: gradient of cost with regard to weight
    """
    dCda = 2 * (func_a1(x, w1, b1) - y)
    z1 = w1 * x + b1
    dadz = 1/np.cosh(z1) ** 2
    dzdw = w1

    return dCda * dadz * dzdw

def dCdb_two_node_neural_network(x, w1, b1, y):
    """
    Cost gradient with regard to bias for a two node neural network.

    Args:
        x: input
        y: desired output
        w1: weight of connection between neuron (0) and neuron (1)
        b1: bias of neuron (1)

    Returns:
        dCdw: gradient of cost with regard to bias
    """
    dCda = 2 * (func_a1(x, w1, b1) - y)
    z1 = w1 * x + b1
    dadz = 1/np.cosh(z1) ** 2
    dzdb = 1

    return dCda * dadz * dzdb


def func_cost(x, W, b, y):
    """
    Cost of training a node neural network.

    Args:
        x: activation of input neurons
        y: desired outputs
        W: weight matrix
        b: bias vector

    Returns:
        cost: training cost
    """
    d = single_layer_neural_network_no_hidden_layer(x, W, b) - y
    cost = d @ d
    return cost


if __name__ == '__main__':

    np.set_printoptions(precision=3)

    a0 = np.array([0.3, 0.4, 0.1])
    W = np.array([[-2, 4, -1], [6, 0, -3]])
    b = np.array([0.1, -2.5])
    a1 = single_layer_neural_network_no_hidden_layer(a0, W, b)
    print(f"Output of a single layer neural network with no hidden layer:")
    print(f"Input a0 = {a0}")
    print(f"Weights W = {W}")
    print(f"biases b = {b}")
    print(f"result: {a1}")
    print()

    # example of training NOT function
    x = 1 # input
    y = 0 # desired output
    w = 1.3
    b = -0.1
    a1 = func_a1(x, w, b).round(3)
    cost = cost_two_node_neural_network(a1, y).round(3)
    print(f"Cost of training a NOT function:")
    print(f"Input x = {x}")
    print(f"Desired output y = {y}")
    print(f"Weight w = {w}")
    print(f"Bias b = {b}")
    print(f"Output as calculated by the feed-forward equation: {a1}")
    print(f"Cost: {cost}")
    print()

    x = 0 # input
    y = 1 # desired output
    w = 1.3
    b = -0.1
    a1 = func_a1(x, w, b).round(3)
    cost = cost_two_node_neural_network(a1, y).round(3)
    print(f"Input x = {x}")
    print(f"Desired output y = {y}")
    print(f"Weight w = {w}")
    print(f"Bias b = {b}")
    print(f"Output as calculated by the feed-forward equation: {a1}")
    print(f"Cost: {cost}")
    print()

    w1 = 2.3
    b1 = -1.2
    x = 0
    y = 1
    print(f"Gradient of cost with regard to bias for a two node neural network:")
    print(f"Weight w1 = {w1}")
    print(f"Bias b1 = {b1}")
    print(f"Input x = {x}")
    print(f"Desired output y ={y}")
    print(f"dCdb: {dCdb_two_node_neural_network(x, w1, b1, y)}")
    print()

    x = np.array([0.7, 0.6, 0.2])
    y = np.array([0.9, 0.6])
    W = np.array(
        [
            [-0.94529712, -0.2667356, -0.91219181],
            [2.05529992, 1.21797092, 0.22914497],
        ]
    )
    b = np.array([0.61273249, 1.6422662])
    print(f"Cost function for a neural network with no hidden layer:")
    print(f"Inputs x = {x}")
    print(f"Desired outputs y = {y}")
    print(f"Weight matrix W = {W}")
    print(f"Bias vector b = {b}")
    cost = func_cost(x, W, b, y).round(1)
    print(f"Cost: {cost}")
    print()
