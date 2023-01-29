from typing import List, Tuple
import math
from scipy.optimize import fsolve
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

def plot_optimization(func, bounds: Tuple[float, float], x):
    l, r = bounds
    x_values = np.linspace(l, r, 100)
    y_values = list(map(func, x_values))
    # fig = px.line(x=x_values, y=y_values, labels={'x': 'x', 'y': 'e^(x-5) + e^(5-x)'})
    fig = px.line(x=x_values, y=y_values, labels={'x': 'x', 'y': 'f(x)'})
    fig.update_traces(line=dict(width=4))
    y_iter = list(map(func, x))
    fig.add_trace(go.Scatter(
        x=x,
        y=y_iter,
        mode="markers+text",
        name='iteration',
        marker_size=15,
        text=[str(i+1) for i in range(len(x))])
    )
    fig.add_trace(go.Scatter(
        x=[x[-1]],
        y=[y_iter[-1]],
        mode="markers",
        name='extremum',
        marker_size=15,
        fillcolor="blue")
    )
    fig.update_traces(textposition='top center')
    fig.show()
    
def golden_section_method(func, bounds: Tuple[float, float], eps: float, plot=False) -> float:
    l, r = bounds
    phi = (1 + math.sqrt(5)) / 2
    x1, x2 = r - (r-l) / phi, l + (r-l) / phi
    num_calc, interval_change, max_iter = 2, 0, 100
    print('Golden Section Method:')
    print(f'f(x) = e^(x-5) + e^(5-x), l = {l}, r = {r}')
    x_k = (l+r) / 2
    x = np.array([x_k])
    print(f'Iteration {1}: length of interval = {(r-l):.5f}, x = {x_k:.5f}, f(x) = {func(x_k):.5f}')
    fx1, fx2 = func(x1), func(x2)
    
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
        
        interval_change += 1
        x_k = (l + r) / 2
        x = np.append(x, x_k)
        num_calc += 1
        print(f'Iteration {i+2}: length of interval = {(r - l):.5f}, x = {x_k:.5f}, f(x) = {func(x_k):.5f}')
        
        if (r - l < eps):
            break

    print(f'Result: x = {x_k:.5f}, f(x) = {func(x_k):.5f}')
    print(f'Number of calculations: {num_calc}')
    
    if plot:
        plot_optimization(func, bounds, x)
        
    return (l + r) / 2, interval_change, num_calc
    
def fib(n: int) -> int:
    a, b = 0, 1
    i = 2
    while i != n:
        c = a + b
        a, b = b, c
        i += 1
    return b

def fibonacci_method(func, bounds: Tuple[float, float], eps: float, n: int = 20, plot=False) -> tuple:
    l, r = bounds
    x1, x2 = l + (r-l) * fib(n-2)/fib(n), l + (r-l) * fib(n-1)/fib(n)
    
    print('Fibonacci Method:')
    print(f'f(x) = e^(x-5) + e^(5-x), l = {l}, r = {r}')
    
    x_k = (l + r) / 2
    x = np.array([x_k])
    i, interval_change, num_calc = 1, 0, 2
    print(f'Iteration {i}: length of interval = {(r-l):.5f}, x = {x_k:.5f}, f(x) = {func(x_k):.5f}')
    
    fx1, fx2 = func(x1), func(x2)
    while n != 1 and (r - l) >= eps:
        if fx1 > fx2:
            l = x1
            x1 = x2
            x2 = r - (x1 - l)
            fx1, fx2 = fx2, func(x2)
        else:
            r = x2
            x2 = x1
            x1 = l + (r - x2)
            fx2, fx1 = fx1, func(x1)
            
        interval_change += 1
        x_k = (l + r) / 2
        x = np.append(x, x_k)
        num_calc += 1
        i += 1
        print(f'Iteration {i}: length of interval = {(r - l):.5f}, x = {x_k:.5f}, f(x) = {func(x_k):.5f}')
        n-=1
    
    print(f'Result: x = {x_k:.5f}, f(x) = {func(x_k):.5f}')
    print(f'Number of calculations: {num_calc}')
    
    if plot:
        plot_optimization(func, bounds, x)
    return (x1 + x2) / 2, interval_change, num_calc

def sven_method(func, x0: float) -> Tuple[float, float]:
    step = 0.5
    fl, f, fr = func(x0 - step), func(x0), func(x0 + step)
    
    if fl >= f and f < fr:
        return x0 - step, x0 + step
    if fl < f < fr:
        step = -step
        
    p = 1
    while True:
        f_new = func(x0 + 2**p * step)
        
        if f_new >= f:
            x1 = x0 + 2**(p - 2) * step
            x2 = x0 + 2**(p - 1) * step
            x4 = x0 + 2**p * step
            x3 = (x4 - x2) / 2
            
            return min(x1, x4), max(x1, x4)

        f = f_new
        p += 1
        
f = lambda x: np.exp(x - 5) + np.exp(5 - x)
print(sven_method(f, 2))
golden_section_method(f, (3, 6), 0.01, plot=True)
fibonacci_method(f, (3, 6), 0.01, 20, plot=True)

def calc(method, bounds):
    interval_changes = []
    eps = []
    epsilon = 0.001
    while epsilon <= 0.1:
        interval_changes.append(method(f, bounds, epsilon, plot=False)[1])
        epsilon += 0.001
        eps.append(epsilon)
    return eps, interval_changes

eps1, interval_changes1 = calc(golden_section_method, (3, 6))
eps2, interval_changes2 = calc(fibonacci_method, (3, 6))

fig = go.Figure()
fig.add_trace(go.Scatter(x=eps1, y=interval_changes1, name='golden section'))
fig.add_trace(go.Scatter(x=eps2, y=interval_changes2, name='fibonacci'))
fig.update_layout(xaxis_title='eps', yaxis_title='interval changes')
fig.show()

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