import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint, solve_ivp


e1 = 2.1
y12 = 0.51
y11 = 0.1
y21 = 0.67
e2 = 3
y22 = 0.13

# e1 = 2.1; y12 = 0.51; y11 = 0.1; y21 = 0.67; e2 = 3; y22 = 0.13

# x0 = 1.9
# y0 = 1.6

x0 = 1.9
y0 = 1.6

# e2 = 10

t = np.arange(0, 10, 0.001)

def f(t, y):
    result = [
        e1*y[0] - y12*y[0]*y[1] - y11*(y[0]**2),
        y21*y[0]*y[1] - e2*y[1] - y22*(y[1]**2),
    ]
    return np.array(result)


def main():
    p1 = (e1 / y11, 0)
    print(f'P1 ({e1 / y11:.5f}, 0)')

    t_x = (e1 * y22 + e2 * y12) / (y11 * y22 + y12 * y21)
    t_y = (e1 * y21 - e2 * y11) / (y11 * y22 + y12 * y21)
    # x0, y0 = t_x, t_y
    p2 = (t_x, t_y)
    print(f'P2 ({t_x:.5f}, {t_y:.5f})')

    y1 = odeint(f, [x0, y0], t, tfirst=True)
    
    
    y2 = odeint(f, [21, 0], t, tfirst=True)
    y3 = odeint(f, [2, 0], t, tfirst=True)
    
    # print(f'min={np.min(y1[:, 0]):.5f} max={np.max(y1[:, 0]):.5f}')
    
    ax1 = plt.subplot(2, 2, 1)
    ax2 = plt.subplot(2, 2, 2)
    ax3 = plt.subplot(2, 2, 3)
    
    ax1.set_title('Кількість жертв')
    ax1.set_xlabel('t')
    ax1.set_ylabel('population')
    ax2.set_title('Кількість хижаків')
    ax2.set_xlabel('t')
    ax2.set_ylabel('population')
    # ax3.set_title('Фазові траєкторії')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    
    ax1.plot(t, y1[:, 0], label='1')
    ax2.plot(t, y1[:, 1], label='2')
    
    # ax1.plot(t, y2[:, 0], label='1')
    # ax2.plot(t, y2[:, 1], label='2')
    # ax1.plot(t, y3[:, 0], label='1')
    # ax2.plot(t, y3[:, 1], label='2')
    ax3.plot(y1[:, 0], y1[:, 1], label='3')
    plt.show()

if __name__ == '__main__':
    main()
