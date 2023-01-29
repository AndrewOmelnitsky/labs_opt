import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint, solve_ivp

e1 = 2; y12 = 0.5; y11 = 0.1; y21 = 0.7; e2 = 3; y22 = 0.11
x_init = 2
y_init = 2.7
t = np.arange(0, 20, 0.1)

def f(t, y):
    result = [
        e1*y[0] - y12*y[0]*y[1] - y11*(y[0]**2),
        y21*y[0]*y[1] - e2*y[1] - y22*(y[1]**2),
    ]
    return np.array(result)


def main():
    y1 = odeint(f, [x_init, y_init], t, tfirst=True)
    
    ax1 = plt.subplot(2, 2, 1)
    ax2 = plt.subplot(2, 2, 2)
    ax3 = plt.subplot(2, 2, 3)
    
    ax1.plot(t, y1[:, 0], label='1')
    ax2.plot(t, y1[:, 1], label='2')
    ax3.plot(y1[:, 0], y1[:, 1], label='3')
    plt.show()

if __name__ == '__main__':
    main()
