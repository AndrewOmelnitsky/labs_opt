import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint, solve_ivp

mu = -1; a = 1; b = 1
x_init = 1; y_init = 0
t = np.arange(0, 100, 0.1)

def f(t, y):
    result = [
        y[1],
        mu * (a - y[0]**2) * y[1] - b * y[0],
    ]
    return np.array(result)


def main():
    y1 = odeint(f, [x_init, y_init], t, tfirst=True)
    
    ax1 = plt.subplot(2, 2, 1)
    ax2 = plt.subplot(2, 2, 2)
    ax3 = plt.subplot(2, 2, 3)
    ax3 = plt.subplot(2, 2, 3)
    
    ax1.plot(t, y1[:, 0], label='1')
    ax2.plot(t, y1[:, 1], label='2')
    ax3.plot(y1[:, 0], y1[:, 1], label='3')
    plt.show()

if __name__ == '__main__':
    main()