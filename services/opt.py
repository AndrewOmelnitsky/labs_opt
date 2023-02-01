import copy
import numpy as np
import math
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
        self.is_minimization = True
        self.log_iteration_like_table = True
        
    def __call__(self, func, bounds, *args, function_text=None, **kwargs):
        """Функция func должна быть совместима с использованием numpy"""
        self._log_start(function_text or self.function_text, *bounds)
        
        # Decorate func to count number of calls
        optimiation_func = copy.copy(func)
        optimiation_func = self.optimization_type_function_decorator(optimiation_func)
        optimiation_func = self.function_calls_count(optimiation_func)
        
        result, iterations, *other = self.optimization_method(optimiation_func, bounds, *args, **kwargs)
        
        self._log_result(result, func(result))
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
    
    def optimization_type_function_decorator(self, func):
        def wrap(*args, ignore_call=False, **kwargs):
            result = func(*args, **kwargs)
            if not self.is_minimization:
                return -result
            
            return result
        
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
        if self.log_iteration_like_table:
            print(Fore.GREEN + f'iteration {idx}:' + Style.RESET_ALL, end='')
            print(f'\tL = {L:.5f}', end='')
            print(f'\tx = {x:.5f}', end='')
            print(f'\tf(x) = {y:.5f}', end='')
            print()
            return
            
        print(Fore.GREEN + f'iteration {idx}:' + Style.RESET_ALL)
        print(f'\tL = {L:.5f}')
        print(f'\tx = {x:.5f}')
        print(f'\tf(x) = {y:.5f}')
        
    @_log_decorator
    def _log_start(self, func_text, l, r):
        print(Fore.RED + 'Optimization method: ' + self.method_name + Style.RESET_ALL)
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
    
    def optimization_method(self, func, bounds, eps):
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
