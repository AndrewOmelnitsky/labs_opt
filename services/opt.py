import numpy as np
import matplotlib.pyplot as plt
from colorama import init as colorama_init
colorama_init()
from colorama import Fore, Back, Style

class ABSOptimizationMethod(object):
    method_name = 'optimization method'
    
    def __init__(self):
        self.function_text = 'f(x)'
        self.is_plot = False
        self.is_log = True
        
    def __call__(self, func, bounds, *args, function_text=None, **kwargs):
        """Функция func должна быть совместима с использованием numpy"""
        self._log_start(function_text or self.function_text, *bounds)
        
        # Decorate func to count number of calls
        func = self.function_calls_count(func)
        
        result, iterations, *other = self.optimization_method(func, bounds, *args, **kwargs)
        
        self._log_result(result, func(result, ignore_call=True))
        self._log_number_of_iterations(len(iterations))
        
        if self.is_plot:
            self._plot_it(func, iterations, *bounds)
            
        return result, iterations, *other
    
    def get_interval(self, func, x0, step=0.5):
        """Sven method"""
        
        f_l = func(x0 - step)
        f_r = func(x0 + step)
        f = func(x0)
        
        if f_l >= f and f < f_r:
            return (x0 - step, x0 + step)
        
        if f_l < f < f_r:
            step = -step
            
        p = 1
        while True:
            f_new = func(x0 + 2**p * step)
            
            if f_new >= f:
                a = x0 + 2**(p - 2) * step
                b = x0 + 2**p * step
                
                return tuple(sorted([a, b]))

            f = f_new
            p += 1
    
    def function_calls_count(self, func):
        self.function_calls = 0
        
        def wrap(*args, ignore_call=False, **kwargs):
            if not ignore_call:
                self.function_calls += 1
            return func(*args, **kwargs)
        
        return wrap
    
    @staticmethod
    def _log_decorator(func):
        def wrap(self, *args, **kwargs):
            if not self.is_log:
                return
            
            func(self, *args, **kwargs)
            
        return wrap
    
    @_log_decorator
    def _log_iteration(self, idx, x, y, L):
        print(Fore.GREEN + f'iteration {idx}:' + Style.RESET_ALL)
        print(f'\tL = {L:.5f}')
        print(f'\tx = {x:.5f}')
        print(f'\tf(x) = {y:.5f}')
        
    @_log_decorator
    def _log_start(self, func_text, l, r):
        print(Fore.RED + self.method_name + Style.RESET_ALL)
        print(Fore.GREEN + 'Initial parameters:' + Style.RESET_ALL)
        print(f'\t{func_text}')
        print(f'\tl = {l:.5f}, r = {r:.5f}')
    
    @_log_decorator
    def _log_result(self, x, y):
        print(Fore.GREEN + 'Result:' + Style.RESET_ALL)
        print(f'\tx = {x:.5f}')
        print(f'\tf(x) = {y:.5f}')
        print(Fore.GREEN + 'Function calls:' + Style.RESET_ALL + str(self.function_calls))
        
    @_log_decorator
    def _log_number_of_iterations(self, iters_num):
        print(Fore.GREEN + 'Number of iterations:' + Style.RESET_ALL + str(iters_num))
            
    def _plot_it(self, func, iterations, l, r, steps = 1000):
        color_function = '#1050A8'
        color_iteration = '#FFF702'
        color_extremum = '#F30223'
        
        x = np.linspace(l, r, steps)
        # y = list(map(func, x))
        y = func(x)
        iterations_y = list(map(func, iterations))

        plt.xlabel('x')
        plt.ylabel('f(x)')
        
        # Отображение функции
        plt.plot(x, y, color=color_function, zorder=0)
        
        # Отображение итераций
        plt.scatter(
            iterations, iterations_y,
            label=f'iteration [n={len(iterations)}]',
            color=color_iteration,
            zorder=1,
            s=30,
        )
        
        # Отображени екстремума
        plt.scatter(
            [iterations[-1]], [iterations_y[-1]],
            label=f'extremum [x={iterations[-1]:.5f}, y={iterations_y[-1]:.5f}]',
            color=color_extremum,
            zorder=1,
            s=40,
        )
        
        # Указывание номера итерации над меткой интерации
        for i in range(len(iterations)):
            plt.annotate(str(i + 1), (iterations[i], iterations_y[i]), ha='center', va='bottom')
            
        plt.legend()
        plt.show()
        
    def optimization_method(self):
        ...