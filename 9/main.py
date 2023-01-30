from typing import List, Tuple
import math
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import numpy as np

from services.opt import *

phi = golden_ratio = (1 + math.sqrt(5)) / 2
        
    
class GoldenSectionMethod(ABSOptimizationMethod):
    method_name = 'Golden section method'
    
    def optimization_method(self, func, bounds, eps, max_iter=100):
        l, r = bounds
        iterations = []
        d = (r - l)
        ind = 0
        interval_changes = 0
        
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
            interval_changes += 1
            ind += 1
    
        result = (l + r) / 2
        
        return result, iterations, interval_changes, self.function_calls
    

class FibonacciMethod(ABSOptimizationMethod):
    method_name = 'Fibonacci method'
    
    @staticmethod
    def fib(n):
        a = 0
        b = 1
        
        if n == 0:
            return a
        
        if n == 1:
            return b
        
        for i in range(1, n):
            a, b = b, a + b
            
        return b
    
    def optimization_method(self, func, bounds, eps, n=20):
        fib = self.fib
        l, r = bounds
        iterations = []
        ind = 0
        interval_changes = 0
        
        n = 0
        while fib(n) <= (r - l) / (eps):
            n += 1
        
        n = max(n, 3)
        
        for k in range(n - 2):
            p = (fib(n - k - 1) / fib(n - k))
            x1 = l + (r - l) * (1 - p)
            x2 = l + (r - l) * p
            
            temp_x = (l + r) / 2
            self._log_iteration(ind, temp_x, func(temp_x, ignore_call=True), (r - l))
            
            if func(x1) <= func(x2):
                r = x2
            else:
                l = x1
                
            iterations.append(temp_x)
            interval_changes += 1
            ind += 1
    
        result = (l + r) / 2
        
        return result, iterations, interval_changes, self.function_calls

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
    initial = [0, 0.5, 0.25, 0.1, 0, 0.5, 0.25, 0.1]
    stationary = fsolve(calculate, x0=initial, args=(x,))
    return -G(stationary, x)

# dichotomy_method(profit, (0, 1), 0.01, plot=True)

def compare_methods(method1, method2, eps_range=np.linspace(0.0001, 0.1, 100), method_label1='f1', method_label2='f2'):
    params1 = []
    params2 = []

    def get_params(method, eps):
        result, iters, inter_ch, func_calcs = method(eps)
        return [inter_ch, func_calcs]
        
    for eps in eps_range:
        params1.append(get_params(method1, eps))
        params2.append(get_params(method2, eps))
        
    ax1 = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(1, 2, 2)
    
    
    params1 = np.array(params1)
    params2 = np.array(params2)

    ax1.set_xlabel('epsilon')
    ax1.set_ylabel('Interval changes')
    ax1.plot(eps_range, params1[:, 0], label=method_label1)
    ax1.plot(eps_range, params2[:, 0], label=method_label2)
    
    ax2.set_xlabel('epsilon')
    ax2.set_ylabel('Function calcs')
    ax2.plot(eps_range, params1[:, 1], label=method_label1)
    ax2.plot(eps_range, params2[:, 1], label=method_label2)
    
    ax1.legend()
    ax2.legend()
    plt.show()


def task1():
    f = lambda x:  2 - (1 / (np.log2(x**4 + 4*(x**3) + 29)))
    x0 = -1
    eps = 0.01
    
    f = lambda x: np.exp(x - 5) + np.exp(5 - x)
    x0 = 2
    
    fibonacci_method = FibonacciMethod()
    golden_section_method = GoldenSectionMethod()

    step = 0.5
    interval = fibonacci_method.get_interval(f, x0, step)
    print(f'Sven method interval with {step=}: {interval}')
    fibonacci_method.is_plot = True
    golden_section_method.is_plot = True
    n = 20
    fibonacci_method(f, interval, eps, n=20)
    # golden_section_method(f, interval, eps)
    
    fibonacci_method.is_plot = False
    golden_section_method.is_plot = False
    fibonacci_method.is_log = False
    golden_section_method.is_log = False
    
    compare_methods(
        lambda eps: fibonacci_method(f, interval, eps, n=20),
        lambda eps: golden_section_method(f, interval, eps),
        method_label1='fibonacci_method',
        method_label2='golden_section_method',
    )
    # fibonacci_method(profit, (0, 1), 0.01, n=20)
    
    
def main():
    task1()
    
    
if __name__ == '__main__':
    main()