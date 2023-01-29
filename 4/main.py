import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint, solve_ivp

s = 10; r = 60; b = 1;
x_init = 10; y_init = 10; z_init = 300
t = np.arange(0, 50, 0.01)

def f(t, y):
    result = [
        s * (y[1] - y[0]),
        y[0] * (r - y[2]) - y[1],
        y[0] * y[1] - b * y[2],
    ]
    return np.array(result)


def main():
    y1 = odeint(f, [x_init, y_init, z_init], t, tfirst=True)
    
    ax1 = plt.subplot(2, 2, 1)
    ax2 = plt.subplot(2, 2, 2)
    ax3 = plt.subplot(2, 2, 3)
    ax4 = plt.subplot(2, 2, 4, projection='3d')
    
    ax1.plot(t, y1[:, 0], label='1')
    ax2.plot(t, y1[:, 1], label='2')
    ax3.plot(t, y1[:, 2], label='3')
    ax4.plot(y1[:, 0], y1[:, 1], y1[:, 2], label='4', linewidth=0.5)
    plt.show()
    
    ax4 = plt.subplot(1, 1, 1, projection='3d')
    ax4.plot(y1[:, 0], y1[:, 1], y1[:, 2], label='4', linewidth=0.5)
    plt.show()

if __name__ == '__main__':
    main()
