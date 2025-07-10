import jax.numpy as jnp
from jax import random
import jax


def dense_layer_init(input_size, output_size, key, scale=0.01):
    """Initializes parameters (weights and bias) for a Dense layer."""
    w_key, b_key = random.split(key)
    weights = scale * random.normal(w_key, (input_size, output_size))
    bias = scale * random.normal(b_key, (output_size,))
    return weights, bias


def dense_layer_apply(params, x):
    """Applies a single dense layer transformation to the input.

    Args:
        params (tuple): Contains the layer's weights and biases.
        x (jnp.ndarray): The input tensor.

    Returns:
        jnp.ndarray: The result of the linear transformation.
    """

    weights, bias = params
    result = jnp.dot(x, weights) + bias

    return result


def mlp_init(layer_specs, key):
    """
    Initializes parameters for a multi-layer perceptron (MLP) given a list of layer specifications.
    Args:
        layer_specs (list[tuple[int, int, float]]): A list of tuples where each tuple contains:
            - input_size (int): The dimensionality of the layer's input.
            - output_size (int): The dimensionality of the layer's output.
            - scale (float): A factor to scale the random initialization of the weights.
        key (jax.random.PRNGKey): A JAX PRNG key used for generating pseudorandom values across layers.
    Returns:
        list[dict[str, Any]]: A list of parameter dictionaries for each dense layer.
        Each dictionary typically contains:
            - "W" (Array): The weight matrix for the layer.
            - "b" (Array): The bias vector for the layer.
    Raises:
        ValueError: If the shape specifications are invalid or if any layer_specs entry
            does not match the expected format.
    Examples:
        >>> import jax.random as random
        >>> layer_specs = [(16, 32, 0.1), (32, 10, 0.1)]
        >>> key = random.PRNGKey(0)
        >>> params = mlp_init(layer_specs, key)
        >>> len(params)
        2
    Notes:
        - This function uses random splitting to ensure each layer receives a unique
          PRNG key for parameter initialization.
        - The scale value controls the overall magnitude of the parameter initialization.
    """

    params = []
    for i, spec in enumerate(layer_specs):
        input_size, output_size, scale = spec
        key, layer_key = random.split(key)
        layer_params = dense_layer_init(input_size, output_size, layer_key, scale)
        params.append(layer_params)
    return params


def mlp_apply(params, x):
    """ """

    activations = x

    for i in range(len(params) - 1):  # Iterate through hidden layers
        layer_params = params[i]
        activations = dense_layer_apply(layer_params, activations)
        activations = jax.nn.swish(activations)

    last_layer = params[-1]
    activations = dense_layer_apply(last_layer, activations)
    return activations


def l2_normalize(x):
    """
    L2 normalizes a vector.

    Args:
        x: A JAX array representing the vector.

    Returns:
        The L2 normalized vector.
    """

    norm = jnp.linalg.norm(x)
    return x / jnp.where(norm == 0, 1, norm)  # Avoid division by zero if norm is 0
