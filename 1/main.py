import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint, solve_ivp

e = 1.2
b = 0.35
a = 0.02
d = 0.0001
# e = 0.8; a = 0.014; d = 0.005; b = 0.45;
# e = 2
# t0 = 0;
xs = 200
ys = 200

def f(t, y):
    result = [
        y[0] * (e - a * y[1]),
        y[1] * (d * y[0] - b),
    ]
    return np.array(result)


def main():
    t = np.arange(0, 50, 0.1)
    y1 = odeint(f, [xs, ys], t, tfirst=True)
    y2 = odeint(f, [100, 150], t, tfirst=True)
    y3 = odeint(f, [100, 100], t, tfirst=True)
    
    ax1 = plt.subplot(2, 2, 1)
    ax2 = plt.subplot(2, 2, 2)
    ax3 = plt.subplot(2, 2, 3)
    
    ax1.plot(t, y2[:, 0], label='1')
    ax2.plot(t, y2[:, 1], label='2')
    ax3.plot(y1[:, 0], y1[:, 1], label='3')
    ax3.plot(y2[:, 0], y2[:, 1], label='3')
    ax3.plot(y3[:, 0], y3[:, 1], label='3')
    plt.show()


if __name__ == '__main__':
    main()
