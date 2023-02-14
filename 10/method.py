import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import numpy as np
from typing import List, Tuple
import math

from services.opt import *


def part_diff(function, point, diff_id, eps=1e-9):
    d_point = point.copy()
    d_point[diff_id] = point[diff_id] + eps
    return (function(d_point) - function(point)) / eps


def grad_in_point(function, point):
    grad = []
    for i in range(len(point)):
        grad.append(part_diff(function, point, i))
        
    return np.array(grad)
    

def gradient_descent(func, init, eps, bounds = None, iterations = None, min_cls=DichotomyMethod):
    minimization_method = min_cls()
    minimization_method.is_plot = False
    minimization_method.is_log = False
    
    point = np.array(init, dtype=float)
    
    while True:
        if iterations is not None:
            iterations.append(point.copy())
        
        n_grad = grad_in_point(func, point)
        # grad = n_grad
        grad = n_grad / np.linalg.norm(n_grad, ord=1)
        
        if bounds is not None:
            for i, (start, end) in enumerate(bounds):
                if abs(point[i] - start) < eps:
                    grad[i] = min(grad[i], 0)
                    n_grad[i] = min(n_grad[i], 0)
                elif abs(point[i] - end) < eps:
                    grad[i] = max(grad[i], 0)
                    n_grad[i] = max(n_grad[i], 0)
            
        f = lambda alpha: func(point - (alpha * grad))
        
        sub_eps = 1e-6
        search_bounds = minimization_method.get_interval(f, 0, min(sub_eps / np.max(np.abs(n_grad)), sub_eps))
        alpha, *_ = minimization_method(f, search_bounds, sub_eps)
        step = grad * (-alpha)
        
        if np.sum(step**2)**0.5 < eps:
            return point
    
        point += step
        
        if bounds is not None:
            for i, (start, end) in enumerate(bounds):
                if point[i] < start:
                    point[i] = start
                elif point[i] > end:
                    point[i] = end
                    

def rotate(coords, angle):
    rot = [[math.cos(angle), -math.sin(angle)],[math.sin(angle), math.cos(angle)]]
    return np.matmul(coords, rot)


def get_rotated_matrix(angle):
    def rot_decorator(func):
        def wrap(x, y, *args, **kwargs):
            x, y = rotate(np.array([x, y]), angle)
            return func(x, y, *args, **kwargs)
        return wrap
    return rot_decorator
 
            
def test_func(x, y, A, b):
    return A[0][0] * x**2 + (A[0][1] + A[1][0]) * x * y + A[1][1] * y**2 + b[0] * x + b[1] * y


def test_8():
    x0 = 1
    y0 = 1
    init_p = np.array([x0, y0])
    eps = 0.001
    angles = np.linspace(0, math.pi*2, 90)
    # angles = np.linspace(-math.pi/4, math.pi*2 + math.pi/4, 90)
    # angles = np.linspace(-math.pi*2, math.pi*2, 90)

    # for report
    mins = lambda n: math.pi * n + math.pi/4*3
    maxs = lambda n: math.pi * n + math.pi/4
    
    
    es = [1, 10, 100]
    for e in es:
        A = [[1, 0], [0, e]]
        b = [0, 0]
        
        y = []
        for angle in angles:
            func = get_rotated_matrix(angle)(test_func)
            target_func = lambda p: func(*p, A, b)
            
            grad_iters = []
            gradient_descent(target_func, init_p, eps, iterations=grad_iters)
            
            y.append(len(grad_iters) - 1)
            
        plt.plot(angles/math.pi * 180, y, label=f'e={e}')

    plt.xlabel('Кут')
    plt.ylabel('Кількість ітерацій')
    plt.legend()
    plt.show()
    
    
# test_8()