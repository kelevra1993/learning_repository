import os

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import jax
import jax.numpy as jnp


def multiplication_function(a: float, x: float, b: float) -> float:
    return a * x ** 2 + b


multiplication_function_gradient = jax.grad(multiplication_function)
a = 2.
x = 4.
b = 1.

print(multiplication_function_gradient(a, b, x))
