from typing import List, Tuple
from collections.abc import Iterable
import math
from scipy.optimize import fsolve
import numpy as np

import math
import matplotlib.pyplot as plt
# from scipy.optimize import fsolve
import numpy as np
from services.opt import *

phi = golden_ratio = (1 + math.sqrt(5)) / 2
        
    
class GoldenSectionMethod(ABSOptimizationMethod):
    method_name = 'Golden section method'
    
    def optimization_method_ex(self, func, bounds, eps, max_iter=100):
        l, r = bounds
        iterations = []
        
        x1 = r - (r-l) / phi
        x2 = l + (r-l) / phi

        x_k = (l+r) / 2
        fx1 = func(x1)
        fx2 = func(x2)
        self._log_iteration(0, x_k, func(x_k, ignore_call=True), (r - l))
        iterations.append(x_k)
        
        for i in range(max_iter):
            if fx1 < fx2:
                r = x2
                x2 = x1
                x1 = l + (r - x2)
                fx2, fx1 = fx1, func(x1)
            else:
                l = x1
                x1 = x2
                x2 = r - (x1 - l)
                fx1, fx2 = fx2, func(x2)
            
            x_k = (l + r) / 2
            
            self._log_iteration(i+1, x_k, func(x_k, ignore_call=True), (r - l))
            iterations.append(x_k)
            
            if (r - l < 2 * eps):
                break
    
        result = (l + r) / 2
        
        return result, iterations
    
    def optimization_method(self, func, bounds, eps, max_iter=100):
        l, r = bounds
        iterations = []
        d = (r - l)
        ind = 0
        
        while (r - l) >= eps:
            d = d / phi
            x1 = r - d
            x2 = l + d
            
            temp_x = (l + r) / 2
            self._log_iteration(ind, temp_x, func(temp_x, ignore_call=True), (r - l))
            
            if func(x1) <= func(x2):
                r = x2
            else:
                l = x1
                
            iterations.append(temp_x)
            ind += 1
    
        result = (l + r) / 2
        
        return result, iterations, 1, 2, 3


# TODO: 
# def golden_section_method(func, bounds: Tuple[float, float], eps: float, plot=False) -> float:
#     l, r = bounds
#     phi = (1 + math.sqrt(5)) / 2
#     x1, x2 = r - (r-l) / phi, l + (r-l) / phi
#     num_calc, max_iter = 2, 100
#     print('Golden Section Method:')
#     print(f'f(x) = e^(x-5) + e^(5-x), l = {l}, r = {r}')
#     x_k = (l+r) / 2
#     x = np.array([x_k])
#     print(f'Iteration {1}: length of interval = {(r-l):.5f}, x = {x_k:.5f}, f(x) = {func(x_k):.5f}')
#     fx1, fx2 = func(x1), func(x2)
    
#     for i in range(max_iter):
#         if fx1 < fx2:
#             r = x2
#             x2 = x1
#             x1 = l + (r - x2)
#             fx2, fx1 = fx1, func(x1)
#         else:
#             l = x1
#             x1 = x2
#             x2 = r - (x1 - l)
#             fx1, fx2 = fx2, func(x2)
        
#         x_k = (l + r) / 2
#         x = np.append(x, x_k)
#         num_calc += 1
#         print(f'Iteration {i+2}: length of interval = {(r - l):.5f}, x = {x_k:.5f}, f(x) = {func(x_k):.5f}')
        
#         if (r - l < 2 * eps):
#             break

#     print(f'Result: x = {x_k:.5f}, f(x) = {func(x_k):.5f}')
#     print(f'Number of calculations: {num_calc}')
    
#     if plot:
#         plot_optimization(func, bounds, x)
        
#     return (l + r) / 2
        
# f = lambda x: np.exp(x - 5) + np.exp(5 - x)
# # dichotomy_method(f, (3, 6), 0.01, plot=True)
# golden_section_method(f, (3, 6), 0.01, plot=True)


aph = 0.5
bt = 1.5
gm = 1.5
dt = 0.1
nu = 5
mu = 20
lmd = 20
ro = 10
A0, L0, D0 = 1, 1, 1
tau = 0.4
sigma = 0.2
theta = (1 + aph * (bt - 1)) ** (-1)
left = 0
right = 600
resolution = 100

def L1(data: List[float]):
    return data[3] * ((1 - aph) * A0 * data[1] / data[2]) ** (1 / aph)

def Q1(data: List[float]):
    return A0 * data[3] ** aph * L1(data) ** (1 - aph)

def D1(data: List[float]):
    return D0 * math.exp(-bt * data[1]) * data[5] / (data[1] + data[5])

def S1(data: List[float]):
    return L0 * (1 - math.exp(-gm * data[2])) * data[2] / (data[2] + data[6])

def I1(data: List[float], tau):
    return (1 - tau) * (1 - theta) * data[0]

def G1(data: List[float], tau):
    return (1 - tau) * theta * data[0]

def L2(data: List[float]):
    return data[7] * ((1 - aph) * A0 * data[5] / data[6]) ** (1 / aph)

def Q2(data: List[float]):
    return A0 * data[7] ** aph * L2(data) ** (1 - aph)

def D2(data: List[float]):
    return D0 * math.exp(-bt * data[5]) * data[1] / (data[1] + data[5])

def S2(data: List[float]):
    return L0 * (1 - math.exp(-gm * data[6])) * data[6] / (data[2] + data[6])

def I2(data: List[float]):
    return (1 - theta) * data[4]

def G2(data: List[float]):
    return theta * data[4]

def T(data: List[float], tau):
    return tau * data[0]

def G(data: List[float], tau):
    return (1 - sigma) * tau * data[0]

def calculate(data: List[float], *tau):
    result = [0] * 8
    result[0] = (data[1] * min(Q1(data), D1(data)) - data[2] * min(L1(data), S1(data)) - data[0]) / nu
    result[1] = (D1(data) - Q1(data)) / mu
    result[2] = (L1(data) - S1(data)) / lmd
    result[3] = -dt * data[3] + I1(data, tau[0])
    result[4] = (math.exp(-ro * sigma * T(data, tau[0])) * data[5] * min(Q2(data), D2(data)) - data[6] * min(L2(data), S2(data)) - data[4]) / nu
    result[5] = (D2(data) - Q2(data)) / mu
    result[6] = (L2(data) - S2(data)) / lmd
    result[7] = -dt * data[7] + I2(data)
    return result

def profit(x):
    if isinstance(x, Iterable):
        return list(map(profit, list(x)))
    
    initial = [0, 0.5, 0.25, 0.1, 0, 0.5, 0.25, 0.1]
    stationary = fsolve(calculate, x0=initial, args=(x,))
    return -G(stationary, x)

# dichotomy_method(profit, (0, 1), 0.01, plot=True)

def task1():
    f = lambda x:  2 - (1 / (np.log2(x**4 + 4*(x**3) + 29)))
    x0 = -1
    eps = 0.01
    
    f = lambda x: np.exp(x - 5) + np.exp(5 - x)
    x0 = 2

    golden_section_method = GoldenSectionMethod()

    step = 0.5
    interval = golden_section_method.get_interval(f, x0, step)
    print(f'Sven method interval with {step=}: {interval}')
    golden_section_method.is_plot = True
    golden_section_method(f, interval, eps)
    
    # golden_section_method(profit, (0, 1), 0.01)
    
    
def main():
    task1()
    
    
if __name__ == '__main__':
    main()