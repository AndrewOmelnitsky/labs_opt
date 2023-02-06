import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint, solve_ivp

s = 10
r = 60
b = 3
x_0 = 10
y_0 = 10
z_0 = 10
t = np.arange(0, 10, 0.001)

s_list = [0, 10, 20, 30, 40, 50]
r_list = [0, 10, 20, 30, 40, 50]

# s_list = [0, 10]
# r_list = [0, 10]

s_list = [0]
r_list = [10, 50]
is_save = True


def f(t, y):
    result = [
        s * (y[1] - y[0]),
        y[0] * (r - y[2]) - y[1],
        y[0] * y[1] - b * y[2],
    ]
    return np.array(result)


def latex_ex_section_pattern(idx, sigam, r, b, links):
    link1, link2, link3 = links
    
    sparam = fr'\(\sigma = {sigam}, r = {r}, b = {b}\)'
    s00 = r'\newpage'
    s0 = r'\subsection{Приклад ' + str(idx) + '}'
    s1 = fr'Розглянемо систему за таких початкових умов: {sparam}.\\'
    s2 = r'\begin{figure}[h!]'
    s3 = r'\centering'
    s4 = r'\includegraphics[width=\textwidth]{lab4/' + link1 + '}'
    s5 = r'\caption{Приклад. (' + sparam + ')}'
    s6 = r'\label{fig:ex1}'
    s7 = r'\end{figure}\\'
    
    l1 = [s00, s0, s1, s2, s3, s4, s5, s6, s7]
    
    s2 = r'\begin{figure}[h!]'
    s3 = r'\centering'
    s4 = r'\includegraphics[width=\textwidth]{lab4/' + link2 + '}'
    s5 = r'\caption{У фазовому просторі. (' + sparam + ')}'
    s6 = r'\label{fig:ex1}'
    s7 = r'\end{figure}\\'
    
    l2 = [s2, s3, s4, s5, s6, s7]
    
    s2 = r'\begin{figure}[h!]'
    s3 = r'\centering'
    s4 = r'\includegraphics[width=\textwidth]{lab4/' + link3 + '}'
    s5 = r'\caption{У просторі. (' + sparam + ')}'
    s6 = r'\label{fig:ex1}'
    s7 = r'\end{figure}\\'
    
    l3 = [s2, s3, s4, s5, s6, s7]
    l = [*l1, *l2, *l3]
    
    return '\n'.join(l)


def set_title(title: str):
    plt.get_current_fig_manager().set_window_title(title)


def main():
    result_latex_text = ''
    idx = 0
    global s, r
    for r in r_list:
        for s in s_list:
            strfy = lambda x: f'_{abs(x)}' if x < 0 else str(x)
            link = f'images/r_{strfy(r)}___sigma_{strfy(s)}.png'
            link1 = f'images/1__r_{strfy(r)}___sigma_{strfy(s)}.png'
            link2 = f'images/2__r_{strfy(r)}___sigma_{strfy(s)}.png'
            link3 = f'images/3__r_{strfy(r)}___sigma_{strfy(s)}.png'
            
            
            y1 = odeint(f, [x_0, y_0, z_0], t, tfirst=True)
            
            
            
            
            plt.figure(figsize=(16, 4))
            set_title(link)
            ax1 = plt.subplot(1, 3, 1)
            ax2 = plt.subplot(1, 3, 2)
            ax3 = plt.subplot(1, 3, 3)
            
            ax1.set_title('x(t)')
            ax1.set_xlabel('t')
            ax1.set_ylabel('x(t)')
            
            ax2.set_title('y(t)')
            ax2.set_xlabel('t')
            ax2.set_ylabel('y(t)')
            
            ax3.set_title('z(t)')
            ax3.set_xlabel('t')
            ax3.set_ylabel('z(t)')
            
            ax1.plot(t, y1[:, 0], label='1')
            ax2.plot(t, y1[:, 1], label='2')
            ax3.plot(t, y1[:, 2], label='3')
            if is_save:
                plt.savefig(link1)
            plt.show()
            
            
            
            
            plt.figure(figsize=(16, 4))
            set_title(link)
            ax1 = plt.subplot(1, 3, 1)
            ax2 = plt.subplot(1, 3, 2)
            ax3 = plt.subplot(1, 3, 3)
            
            ax1.set_title('Проекція xy')
            ax1.set_xlabel('x')
            ax1.set_ylabel('y')
            
            ax2.set_title('Проекція xz')
            ax2.set_xlabel('x')
            ax2.set_ylabel('z')
            
            ax3.set_title('Проекція yz')
            ax3.set_xlabel('y')
            ax3.set_ylabel('z')
            
            ax1.plot(y1[:, 0], y1[:, 1], label='1')
            ax2.plot(y1[:, 0], y1[:, 2], label='2')
            ax3.plot(y1[:, 1], y1[:, 2], label='3')
            if is_save:
                plt.savefig(link2)
            plt.show()
            
            
            
            
            plt.figure(figsize=(10, 10))
            set_title(link)
            ax1 = plt.subplot(1, 1, 1, projection='3d')
            
            ax1.set_title('Фазові траєкторії')
            ax1.set_xlabel('x(t)')
            ax1.set_ylabel('y(t)')
            ax1.set_zlabel('z(t)')
            
            ax1.plot(y1[:, 0], y1[:, 1], y1[:, 2], label='4', linewidth=0.5)
            if is_save:
                plt.savefig(link3)
            plt.show()
            
            if input('add? '):
                idx += 1
                result_latex_text += '\n\n\n' + latex_ex_section_pattern(idx, s, r, b, [link1, link2, link3])
                
    print(result_latex_text)


if __name__ == '__main__':
    main()
