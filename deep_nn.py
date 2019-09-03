import numpy as np # type: ignore
from dataclasses import dataclass
from typing import List, Callable

ActivationFunction = Callable[[np.ndarray], np.ndarray]
ActivationGradientFunction = Callable[[np.ndarray], np.ndarray]

@dataclass
class LayerDefinition:
    neurons: int
    activation_fn: ActivationFunction
    activation_gradient_fn: ActivationGradientFunction

@dataclass
class DeepNeuralNetwork:
    layers: List[LayerDefinition]
    cost_fn: Callable[[np.ndarray, np.ndarray], float]
    cost_gradient_fn: Callable[[np.ndarray, np.ndarray], np.ndarray]

@dataclass
class LayerParameters:
    """
    Weights and biases for a single layer.
    """
    W: np.ndarray  # Shape: (neurons in layer, neurons in previous layer)
    B: np.ndarray  # Shape: (neurons in layer, 1)


@dataclass
class LayerActivation:
    """
    Results of a layer post activation
    """
    Z: np.ndarray # Shape: (neurons in layer, examples)
    A: np.ndarray # Shape: (neurons in layer, exaxmples)

@dataclass
class LayerGradients:
    """
    Gradients for a layer resulting from back propagation.
    """
    dA_prev: np.ndarray
    dW: np.ndarray
    db: np.ndarray


def initialize_parameters(seed: int, layers: List[int]) -> List[LayerParameters]:
    """
    Initializes the weights and biases.

    seed: seed value for the random function.
    layers: how many neurons are in each layer, from input to output.
    """
    np.random.seed(seed)
    return [
        LayerParameters(
            W=np.random.randn(layers[i], layers[i-1]) * 0.01,
            B=np.zeros((layers[i], 1))
        ) for i in range(1, len(layers))
    ]


def test_initialize_parameters() -> None:
    parameters = initialize_parameters(1, [2, 3, 1])

    W0 = np.array([[0.01624345, -0.00611756],
                   [-0.00528172, -0.01072969],
                   [0.00865408, -0.02301539]])
    B0 = np.array([[.0, .0]])
    W1 = np.array([[0.01744812, -0.00761207,  0.00319039]])
    B1 = np.array([[.0]])

    # Hidden layer should be 3,2 shape
    assert(np.allclose(parameters[0].W, W0))
    # Hidden layer should be 1,2 shape
    assert(np.allclose(parameters[0].B, B0))
    # Output layer should be 1,3 shape
    assert(np.allclose(parameters[1].W, W1))
    # Output layer should be 1,1 shape
    assert(np.allclose(parameters[1].B, B1))


def propagate_layer_forward(A_prev: np.ndarray, layer: LayerParameters,
                      activation: ActivationFunction) -> LayerActivation:
    """
    Performs forward propagation on a single layer parameter across all 
    examples. 

    A_prev: activation of the previous layer. Shape (neurons on previous layer, number of examples)
    layer: weights and biases for the current layer
    activation: activation function to use
    """
    Z = layer.W.dot(A_prev) + layer.B
    A = activation(Z)
    return LayerActivation(Z=Z, A=A)


def test_propagate_layer_forward() -> None:
    """
    Test with 2 inputs into a layer with 3 neurons and 2 examples
    """

    # Two inputs, two examples
    A_prev = np.array([[2.0, 3.0], [1.0, 2.0]])

    # 3 neurons connected to the two inputs
    layer = LayerParameters(
        W=np.array([[1, 1],
                    [2, 3],
                    [0, 1]]),
        B=np.array([[1],[2],[3]])
    )

    # Activate layer
    activated = propagate_layer_forward(A_prev, layer, lambda Z: Z * 2)

    # Check activation is correct
    Z = np.array([[ 4.,  6.],
       [ 9., 14.],
       [ 4.,  5.]])
    A = np.array([[ 8., 12.],
       [18., 28.],
       [ 8., 10.]])
    assert(np.allclose(activated.Z, Z))
    assert(np.allclose(activated.A, A))

def propagate_forward(X: np.ndarray, layers: List[LayerDefinition], 
        layer_parameters: List[LayerParameters]) -> List[LayerActivation]:
    """
    Calculates the activation for all layers, across all examples.
    """
    A_prev = X
    activations:List[LayerActivation] = []
    for i in range(len(layers)):
        activation = propagate_layer_forward(A_prev, layer_parameters[i], 
                layers[i].activation_fn)
        activations.append(activation)
        A_prev = activation.A
    return activations

def propagate_layer_backward(dA: np.ndarray, layer_parameters: LayerParameters, 
        layer_activation: LayerActivation, previous_layer_activation: LayerActivation, 
        activation_gradient_fn: ActivationGradientFunction) -> LayerGradients:
    """
    Performs backward propagation for a single layer. Returns the gradient for the weights,
    biases, and activation for the layer before.

    dA: activation gradient of the current layer. Shape (neurons, 1)
    activation_gradient_fn: function that returns the derivative of a gradient
    """
    m = layer_activation.Z.shape(1)
    dZ = dA * activation_gradient_fn(layer_activation.Z)
    dW = (1/m) * (dZ.dot(previous_layer_activation.A.T))
    db = (1/m) * np.sum(dZ, axis = 1, keepdims = True)
    dA_prev = layer_parameters.W.T.dot(dZ)
    return LayerGradients(dW=dW, db=db, dA_prev=dA_prev)

def propagate_backward(dA: np.ndarray, layer_parameters: List[LayerParameters], 
        layer_activations: List[LayerActivation], 
        layer_definitions: List[LayerDefinition]) -> List[LayerGradients]
   pass 


def model(X_train, Y_train, initial_parameters, iterations, learning_rate):
    """
    Performs gradient descent and updates the parameters
    over n iterations
    """
    pass


def predict(X: np.ndarray, layers: List[LayerParameters]) -> np.ndarray:
    """
    Runs the model
    """
    pass
