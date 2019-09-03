import numpy as np # type: ignore

def sigmoid(Z: np.ndarray) -> np.ndarray:
    pass


def relu(Z: np.ndarray) -> np.ndarray:
    pass

"""
def sigmoid_gradient(dA: np.ndarray, Z: np.ndarray) -> np.ndarray:
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    assert (dZ.shape == Z.shape)
    return dZ
"""

def sigmoid_gradient(Z: np.ndarray) -> np.ndarray:
    """
    Computes the sigmoid gradient for each item in the vector
    """
    s = 1/(1+np.exp(-Z)) # Regular sigmoid function
    return s * (1-s) # sigmoid derivative

"""
def relu_gradient(dA: np.ndarray, Z: np.ndarray) -> np.ndarray:
    dZ = np.array(dA, copy=True) # just converting dz to a correct object.
    
    # When z <= 0, you should set dz to 0 as well. 
    dZ[Z <= 0] = 0
    
    assert (dZ.shape == Z.shape)
    
    return dZ
"""

def relu_gradient(Z: np.ndarray) -> np.ndarray:
    """
    Computes the ReLU gradient for each item in the vector.
    """
    dZ = np.array(Z, copy=True) # avoid changing the original Z
    
    # When z <= 0, you should set dz to 0 as well. 
    dZ[Z <= 0] = 0
    dZ[Z > 0 ] = 1
    
    assert (dZ.shape == Z.shape)
    
    return dZ


def cross_entropy_cost(Y: np.ndarray, output_layer: np.ndarray) -> float: 
    """
    Calculates the cost (across all examples).

    Y: Expected values. Shape (output neurons, examples)
    output_layer: Actual values (from the output layer). Shape (output neurons, examples)
    """ 
    m = Y.shape[1] 
    cost = (-1/m) * np.sum((Y*np.log(output_layer)) + ((1 - Y)*np.log(1-output_layer)))     

    # Squeeze returns as a number instead of a matrix
    return np.squeeze(cost) # type: ignore

def cross_entropy_cost_gradient():
    pass
