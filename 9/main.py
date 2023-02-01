import matplotlib.pyplot as plt
import numpy as np

from services.opt import *
from services.opt import GoldenSectionMethod, FibonacciMethod
from services.model import EconomicModel, G_by_x
        

def compare_methods(method1, method2, eps_range=np.linspace(0.0001, 0.1, 100), method_label1='f1', method_label2='f2'):
    params1 = []
    params2 = []

    def get_params(method, eps):
        result, iters, inter_ch, func_calcs = method(eps)
        return [inter_ch, func_calcs]
        
    for eps in eps_range:
        params1.append(get_params(method1, eps))
        params2.append(get_params(method2, eps))
        
    # ax1 = plt.subplot(1, 2, 1)
    # ax2 = plt.subplot(1, 2, 2)
    
    # params1 = np.array(params1)
    # params2 = np.array(params2)

    # ax1.set_xlabel('epsilon')
    # ax1.set_ylabel('Interval changes')
    # ax1.plot(eps_range, params1[:, 0], label=method_label1)
    # ax1.plot(eps_range, params2[:, 0], label=method_label2)
    
    # ax2.set_xlabel('epsilon')
    # ax2.set_ylabel('Function calcs')
    # ax2.plot(eps_range, params1[:, 1], label=method_label1)
    # ax2.plot(eps_range, params2[:, 1], label=method_label2)
    
    # ax1.legend()
    
    
    ax1 = plt.subplot(1, 1, 1)
    
    params1 = np.array(params1)
    params2 = np.array(params2)
    
    # ax1.set_xlabel('epsilon')
    # ax1.set_ylabel('Interval changes')
    # ax1.plot(eps_range, params1[:, 0], label=method_label1)
    # ax1.plot(eps_range, params2[:, 0], label=method_label2)
    
    ax1.set_xlabel('epsilon')
    ax1.set_ylabel('Function calcs')
    ax1.plot(eps_range, params1[:, 1], label=method_label1)
    ax1.plot(eps_range, params2[:, 1], label=method_label2)
    
    ax1.legend()
    plt.show()


def task1():
    f = lambda x:  2 - (1 / (np.log2(x**4 + 4*(x**3) + 29)))
    x0 = -1
    eps = 0.01
    
    fibonacci_method = FibonacciMethod()
    golden_section_method = GoldenSectionMethod()
    
    step = 0.001
    interval = fibonacci_method.get_interval(f, x0, step)
    print(f'Sven method interval with {step=}: {interval}')
    fibonacci_method.is_plot = True
    fibonacci_method(f, interval, eps)
    
    fibonacci_method.is_plot = False
    golden_section_method.is_plot = False
    fibonacci_method.is_log = False
    golden_section_method.is_log = False
    
    compare_methods(
        lambda eps: fibonacci_method(f, interval, eps),
        lambda eps: golden_section_method(f, interval, eps),
        method_label1='fibonacci_method',
        method_label2='golden_section_method',
    )


def task2():
    model = EconomicModel()
    profit = lambda x: G_by_x(model, x)
    
    fibonacci_method = FibonacciMethod()
    fibonacci_method.is_minimization = False
    fibonacci_method.is_plot = True
    fibonacci_method.is_log = True
    fibonacci_method(profit, (0, 1), 0.001)
    
    
def main():
    task1()
    task2()
    
    
if __name__ == '__main__':
    main()