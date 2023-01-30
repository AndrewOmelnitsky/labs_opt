import math
import matplotlib.pyplot as plt
# from scipy.optimize import fsolve
import numpy as np
from services.opt import *
        
    
class DichotomyMethod(ABSOptimizationMethod):
    method_name = 'Dichotomy method'
        
    def optimization_method(self, func, bounds, eps):
        l, r = bounds
        
        iterations = []
        idx = 0

        while r - l >= eps:
            temp_x = (l + r) / 2
            x1 = temp_x - eps / 2
            x2 = temp_x + eps / 2
            
            if func(x1) > func(x2):
                l = temp_x
            else:
                r = temp_x
                
            self._log_iteration(idx, temp_x, func(temp_x, ignore_call=True), (r - l))
            
            iterations.append(temp_x)
            idx += 1
            
        result = (l + r) / 2
        iterations.append(result)
        
        # Логирование последней итерации
        self._log_iteration(idx, result, func(result, ignore_call=True), (r - l))
        
        return result, iterations


def task1():
    f = lambda x:  2 - (1 / (np.log2(x**4 + 4*(x**3) + 29)))
    x0 = -1
    eps = 0.01

    dichotomy_method = DichotomyMethod()

    step = 0.5
    interval = dichotomy_method.get_interval(f, x0, step)
    print(f'Sven method interval with {step=}: {interval}')
    dichotomy_method.is_plot = True
    dichotomy_method(f, interval, eps)


def task2():
    ...
    
    
def main():
    task1()
    
    
if __name__ == '__main__':
    main()


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

def L1(data: list[float]):
    return data[3] * ((1 - aph) * A0 * data[1] / data[2]) ** (1 / aph)

def Q1(data: list[float]):
    return A0 * data[3] ** aph * L1(data) ** (1 - aph)

def D1(data: list[float]):
    return D0 * math.exp(-bt * data[1]) * data[5] / (data[1] + data[5])

def S1(data: list[float]):
    return L0 * (1 - math.exp(-gm * data[2])) * data[2] / (data[2] + data[6])

def I1(data: list[float], tau):
    return (1 - tau) * (1 - theta) * data[0]

def G1(data: list[float], tau):
    return (1 - tau) * theta * data[0]

def L2(data: list[float]):
    return data[7] * ((1 - aph) * A0 * data[5] / data[6]) ** (1 / aph)

def Q2(data: list[float]):
    return A0 * data[7] ** aph * L2(data) ** (1 - aph)

def D2(data: list[float]):
    return D0 * math.exp(-bt * data[5]) * data[1] / (data[1] + data[5])

def S2(data: list[float]):
    return L0 * (1 - math.exp(-gm * data[6])) * data[6] / (data[2] + data[6])

def I2(data: list[float]):
    return (1 - theta) * data[4]

def G2(data: list[float]):
    return theta * data[4]

def T(data: list[float], tau):
    return tau * data[0]

def G(data: list[float], tau):
    return (1 - sigma) * tau * data[0]

def calculate(data: list[float], *tau):
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
    initial = [0, 0.5, 0.25, 0.1, 0, 0.5, 0.25, 0.1]
    stationary = fsolve(calculate, x0=initial, args=(x,))
    return -G(stationary, x)

# dichotomy_method(profit, (0, 1), 0.01, plot=True)