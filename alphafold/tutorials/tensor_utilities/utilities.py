"""
The File Contains general tensor utilities that are used throughout the project and make life easier for tensor manipulation
"""

import torch
from torch import tensor



def unsqueeze_tensor(input: tensor, number: int, direction: str) -> tensor:
    """
    todo add documentation
    """
    if direction not in ["left", "right"]:
        print("Warning : direction must either be left or right")
        raise NotImplementedError

    for i in range(number):
        input = input.unsqueeze(dim=-1 if direction == "right" else 0)

    return input
