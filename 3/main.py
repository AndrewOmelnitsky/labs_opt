import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint, solve_ivp

mu = 0
a = 0
b = 3
t = np.arange(0, 100, 0.001)

mu_list = [-10, -1, 0, 1, 10]
a_list = [-10, -1, 0, 1, 10]
mu_list = [1]
a_list = [0]

x0 = 1
y0 = 0


def latex_ex_section_pattern(mu, b, a, link):
    s1 = fr'Розглянемо систему за таких початкових умов: \(\mu = {mu}, \beta = {b}, \alpha = {a}\).\\'
    s2 = r'\begin{figure}[h!]'
    s3 = r'\centering'
    s4 = r'\includegraphics[width=\textwidth]{lab3/' + link + '}'
    s5 = r'\caption{Приклад}'
    s6 = r'\label{fig:ex1}'
    s7 = r'\end{figure}\\'
    s8 = 'Дамо характиристику. har'
    
    l = [s1, s2, s3, s4, s5, s6, s7, s8]
    
    return '\n'.join(l)

def f(t, y):
    result = [
        y[1],
        mu * (a - y[0]**2) * y[1] - b * y[0],
    ]
    return np.array(result)


def set_title(title: str):
    plt.get_current_fig_manager().set_window_title(title)


def main():  
    global mu, a
    for mu in mu_list:
        for a in a_list:
            # strfy = lambda x: f'_{abs(x)}' if x < 0 else str(x)
            # link = f'images/alpha_{strfy(a)}___mu_{strfy(mu)}.png'
            # print()
            # print()
            # print(latex_ex_section_pattern(mu, b, a, link))
            plt.figure(figsize=(10, 3))
            ax1 = plt.subplot(1, 2, 1)
            ax2 = plt.subplot(1, 2, 2)
            
            ax1.set_title('x(t)')
            ax1.set_xlabel('t')
            ax1.set_ylabel('x(t)')
            
            ax2.set_title('Фазові траєкторії')
            ax2.set_xlabel('x(t)')
            ax2.set_ylabel('d/dt (x(t))')
            
            set_title(f'alpha = {a}; mu = {mu}')
            y1 = odeint(f, [x0, y0], t, tfirst=True)
            mu = 3
            y2 = odeint(f, [x0, y0], t, tfirst=True)
            mu = 6
            y3 = odeint(f, [x0, y0], t, tfirst=True)
            
            ax1.plot(t, y1[:, 0], label=f'mu = {1}')
            ax2.plot(y1[:, 0], y1[:, 1], label=f'mu = {1}')
            
            ax1.plot(t, y2[:, 0], label=f'mu = {3}')
            ax2.plot(y2[:, 0], y2[:, 1], label=f'mu = {3}')
            
            ax1.plot(t, y3[:, 0], label=f'mu = {6}')
            ax2.plot(y3[:, 0], y3[:, 1], label=f'mu = {6}')
            
            # try:
            #     plt.savefig(f'alpha_{strfy(a)}___mu_{strfy(mu)}.png')
            # except:
            #     pass
            ax1.legend()
            ax2.legend()
            plt.show()
            # plt.clf()

if __name__ == '__main__':
    main()