import sympy
from sympy import *
import numpy as np
from count_expression import count_grad_func

def count_function_diff(expression):
    x, y = sympy.symbols('x y')
    exp = eval(expression)
    x_d = str(sympy.diff(exp, x))
    y_d = str(sympy.diff(exp, y))
    
    return (x_d, y_d)

def get_grad_func(expression):
    x_d, y_d = count_function_diff(expression)
    # print(x_d, y_d)
    return lambda x, y: count_grad_func(x, y, x_d, y_d)