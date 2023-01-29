import numpy as np
from numpy import *

def count_expression(params: dict, expression: str):
    from numpy import pi
    for ind in params:
        exec(f"{params[ind]['name']}=params[ind]['value']")

    return eval(expression)
