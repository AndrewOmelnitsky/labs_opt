import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from PyQt5 import QtCore, QtWidgets# , QtWebEngineWidgets
from PyQt5.QtGui import QFont
import numpy as np
import plotly.graph_objects as go
from numpy import linalg as LA
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
        
        if np.sum(n_grad**2)**0.5 < eps:
            return point
    
        point += step
        
        if bounds is not None:
            for i, (start, end) in enumerate(bounds):
                if point[i] < start:
                    point[i] = start
                elif point[i] > end:
                    point[i] = end
                    

def draw_path(func, debug_list, bounds, number_of_lines, xtitle='', ytitle=''):
    l, r = bounds
    x = np.linspace(l, r, 30)
    y = np.linspace(l, r, 20)
    X, Y = np.meshgrid(x, y)
    Z = []
    print(len(X[0]), len(Y[:, 0]))
    print(X[0], Y[:, 0])
    for y_t in Y[:, 0]:
        Z.append([])
        for x_t in X[0]:
            Z[-1].append(func([x_t, y_t]))
    Z = np.array(Z)
    minv = np.min(Z)
    maxv = np.max(Z)
    fig = go.Figure(data=
        go.Contour(
            z=Z,
            x=x,
            y=y,
            contours=dict(
                showlabels=True
            ),
            contours_start=minv,
            contours_end=maxv,
            contours_size=(maxv - minv) / number_of_lines,
        )
    )
    x_values = [debug_list[i][0] for i in range(len(debug_list))]
    y_values = [debug_list[i][1] for i in range(len(debug_list))]
    fig.add_trace(go.Scatter(x=x_values, y=y_values, showlegend=False))
    fig.add_trace(go.Scatter(x=[x_values[0]], y=[y_values[0]],
                             name='start', marker=dict(color="Green", size=6), showlegend=True))
    fig.add_trace(go.Scatter(x=[x_values[-1]], y=[y_values[-1]],
                             name=f'extremum ({len(debug_list)-1} iterations)',
    marker=dict(color="Blue", size=6), showlegend=True))
    fig.update_layout(
        legend=dict(
            yanchor="top",
            xanchor="left",
            x=0.01,
            y=0.99,
        ),
        xaxis_title=xtitle,
        yaxis_title=ytitle,
    )
    fig.show()

def rotate_1(mat, theta):
    rot = [[math.cos(theta), math.sin(theta)],[-math.sin(theta), math.cos(theta)]]
    return np.matmul(np.transpose(rot), np.matmul(mat, rot))


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
            
# def get_start_position(height, A, b):
#     D = b[0]**2 + 4 * A[0][0] * height
#     return (-b[0] + math.sqrt(D)) / (2 * A[0][0])








# aph = 0.5
# bt = 1.5
# gm = 1.5
# dt = 0.1
# nu = 5.0
# mu = 20.0
# lmd = 20.0
# ro = 10.0
# A0 = 1.0
# L0 = 1.0
# D0 = 1.0
# theta = (1 + aph * (bt - 1)) ** (-1)
# def L1(data):
# return data[3] * ((1 - aph) * A0 * data[1] / data[2]) ** (1 / aph)
# def Q1(data):
# return A0 * data[3] ** aph * L1(data) ** (1 - aph)
# def D1(data):
# return D0 * math.exp(-bt * data[1]) * data[5] / (data[1] + data[5])
# def S1(data):
# return L0 * (1 - math.exp(-gm * data[2])) * data[2] / (data[2] + data[6]);
# def I1(data, tau):
# # print(tau)
# return (1 - tau) * (1 - theta) * data[0]
# def G1(data, tau):
# return (1 - tau) * theta * data[0]
# def L2(data):
# return data[7] * ((1 - aph) * A0 * data[5] / data[6]) ** (1 / aph)
# def Q2(data):
# return A0 * data[7] ** aph * L2(data) ** (1 - aph)
# def D2(data):
# return D0 * math.exp(-bt * data[5]) * data[1] / (data[1] + data[5])
# def S2(data):
# return L0 * (1 - math.exp(-gm * data[6])) * data[6] / (data[2] + data[6])
# def I2(data):
# return (1 - theta) * data[4]
# def G2(data):
# return theta * data[4]
# #
# def T(data, tau):
# return tau * data[0]
# def G(data, tau, sigma):
# return (1 - sigma) * tau * data[0]
# def calculate(data, *args):
# result = [0, 0, 0, 0, 0, 0, 0, 0]
# result[0] = (data[1] * min(Q1(data), D1(data)) - data[2] * min(L1(data), S1(data)) - data[0]) /
# nu
# result[1] = (D1(data) - Q1(data)) / mu
# result[2] = (L1(data) - S1(data)) / lmd
# result[3] = -dt * data[3] + I1(data, args[0])
# result[4] = (math.exp(-ro * args[1] * T(data, args[0])) * data[5] * min(Q2(data), D2(data)) -
# data[6] * min(L2(data), S2(data)) - data[4]) / nu
# result[5] = (D2(data) - Q2(data)) / mu
# result[6] = (L2(data) - S2(data)) / lmd
# result[7] = -dt * data[7] + I2(data)
# return result
# def profit(tau, sigma):
# initial = [0.0, 0.5, 0.25, 0.1,
# 0.0, 0.5, 0.25, 0.1]
# stationary = fsolve(calculate, x0=initial, args=(tau, sigma))
# return -G(stationary, tau, sigma)









#
x0, y0 = 3, 3
eps = 0.001
e = 100
A = [[2, 1], [1, 4]]
b = [0, 0]

A = [[1, 0], [0, e]]
height = 10
# x0 = get_start_position(height, A, b)

bounds = [[-6, 6], [-6, 6]]

@get_rotated_matrix(math.pi / 4)
def test_func(x, y, A, b):
    return A[0][0] * x**2 + (A[0][1] + A[1][0]) * x * y + A[1][1] * y**2 + b[0] * x + b[1] * y
    

target_func = lambda p: test_func(*p, A, b)
init_p = np.array([x0, y0])


grad_iters = []

x, y = gradient_descent(target_func, init_p, eps, iterations=grad_iters)

print("Extremum: ", x, y)
draw_path(target_func, grad_iters, bounds[0], 10)
print("f(x,y) = ", target_func([x, y]))

axis_titles = ["x", "y"]

sq = 5
bounds = (-sq, -sq), (sq, sq)

resolution = 50
ncontours = 15

method_names = ["Gradient Descent"]

# draw(to_minimise, axis_titles, bounds, resolution, ncontours, [debug_list], method_names)
start, end = 0, math.pi / 2
steps = 90
eps = 0.001

epsilons = [1, 5, 10, 20, 100]

# fig = go.Figure()
# for epsilon in epsilons:
#     A = [[1, 0], [0, epsilon]]
#     b = [0, 0]
#     angles = np.linspace(start, end, steps)
#     height = epsilon

#     stats = []
#     for theta in angles:
#         rotA, rotb = rotate(A, theta), b
#         debug_list = []

#         to_minimise = get_target_function(rotA, rotb)
#         x0 = get_start_position(height, rotA, rotb)

#         x, y = gradient_descent(to_minimise, (x0, 0), eps, None, debug_list)

#     stats.append(len(debug_list) - 1)
#     theta = angles[len(angles) // 2 + 1]
#     rotA, rotb = rotate(A, theta), b
#     debug_list = []
#     to_minimise = get_target_function(rotA, rotb)
#     x0 = get_start_position(height, rotA, rotb)
#     x, y = gradient_descent(to_minimise, (x0, 0), eps, None, debug_list)
#     axis_titles = ["x","y"]
#     sq = 10
#     bounds = (-sq, -sq), (sq, sq)
#     res = 50
#     draw_path(to_minimise, debug_list, (-4, 4), 10)


#     # fig.add_trace(go.Scatter(x=angles, y=stats,
#     # mode='lines',
#     # name=f'e = {epsilon}'))
#     #
#     # fig.update_layout(
#     # xaxis_title="Angle",
#     # yaxis_title="Number of iterations",
#     # )
#     #

# fig.show()  
#







# to_minimise = profit
# x0, y0 = 0.5, 0.5
# eps = 0.01
# bounds_vars = ((0.0, 1.0), (0.0, 1.0))
# debug_list = []
# x, y = gradient_descent(to_minimise, (x0, y0), eps, bounds_vars, debug_list)
# print("x = ", x, "y = ", y)
# print("f(x,y) = ", to_minimise(x, y))
# axis_titles = ["tau", "sigma"]
# sq = 1
# bounds = (0.0, 0.0), (1.0, 1.0)
# resolution = 50
# ncontours = 15
# method_names = ["Gradient Descent"]
# print(debug_list)
# f = np.vectorize(to_minimise)
# draw_path(f, debug_list, (0, 1), 10, '$\\tau$', '$\sigma$')