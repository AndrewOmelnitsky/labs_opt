import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint, solve_ivp

e = 1.2
b = 0.35
a = 0.02
d = 0.0001

# e = 0.01

# a = 0.1

# b = 0.8

x0 = 100
y0 = 0

# x0 = 500
# y0 = 100

# x0 = 3500
# y0 = 60

# x0 = 3000
# x0 = 4000

# y0 = 55
# y0 = 65


# x0 = 300
# y0 = 0


t_step = 0.1
t_start = 0
t_end = 100
t = np.arange(t_start, t_end, t_step)


def f(t, y):
    result = [
        y[0] * e - a * y[0] *y[1],
        d * y[1] * y[0] - y[1] * b,
    ]
    return np.array(result)


def main():
    y1 = odeint(f, [x0, y0], t, tfirst=True)
    # print(y1)

    y2 = odeint(f, [100, 200], t, tfirst=True)
    y3 = odeint(f, [100, 100], t, tfirst=True)
    
    print('Кількість жертв:')
    print(f'min={np.min(y1[:, 0]):.5f} max={np.max(y1[:, 0]):.5f}')
    print('Кількість хижаків:')
    print(f'min={np.min(y1[:, 1]):.5f} max={np.max(y1[:, 1]):.5f}')
    
    print('')
    
    # y1_max0 = np.max(y1[:, 0])
    # print(y1_max0)
    # idx = 0
    # last = 0
    # lasts = []
    # for i, item in enumerate(t):
    #     if abs(y1[i][0] - y1_max0) < 23:
    #         idx += 1
    #         print(f'\\par{idx} {item:.1f} {y1[i][0]:.5f}\\\\')
    #         # print(item - last)
    #         lasts.append(item - last)
    #         last = item
            
    # lasts.pop(0)
    # print(lasts)
    # print(f'T = {sum(lasts) / len(lasts)}')
    # (11.4 + 11.4 + 11.5 + 11.4) / 4
    # 11.42
    
    # print('')
            
    # x1_max0 = np.max(y1[:, 1])
    # print(x1_max0)
    # idx = 0
    # lasts = []
    # for i, item in enumerate(t):
    #     if abs(y1[i][1] - x1_max0) < 0.1:
    #         idx += 1
    #         print(f'\\par{idx} {item:.1f} {y1[i][1]:.5f}\\\\')
    #         # print(item - last)
    #         lasts.append(item - last)
    #         last = item
            
    # lasts.pop(0)
    # print(lasts)
    # print(f'T = {sum(lasts) / len(lasts)}')
            
    11.4 + 11.5 + 11.4 + 11.5
    
    ax1 = plt.subplot(2, 2, 1)
    ax2 = plt.subplot(2, 2, 2)
    ax3 = plt.subplot(2, 2, 3)
    
    ax1.set_title('Кількість жертв')
    ax1.set_xlabel('t')
    ax1.set_ylabel('population')
    ax2.set_title('Кількість хижаків')
    ax2.set_xlabel('t')
    ax2.set_ylabel('population')
    ax1.plot(t, y1[:, 0], linewidth=0.5)
    ax2.plot(t, y1[:, 1], linewidth=0.5)
    
    # ax3.set_title('Фазові траєкторії')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.plot(y1[:, 0], y1[:, 1], label='3', linewidth=0.5)
    ax3.plot(y2[:, 0], y2[:, 1], label='3', linewidth=0.5)
    ax3.plot(y3[:, 0], y3[:, 1], label='3', linewidth=0.5)
    plt.show()


if __name__ == '__main__':
    main()
