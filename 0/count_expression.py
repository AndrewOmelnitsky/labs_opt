import numpy as np
from numpy import *

def count_expression(params: dict, expression: str):
    from numpy import pi
    for ind in params:
        exec(f"{params[ind]['name']}=params[ind]['value']")
    
    return eval(expression)

def count_grad_func(x, y, x_d, y_d):
    return np.array([-eval(x_d), -eval(y_d)])