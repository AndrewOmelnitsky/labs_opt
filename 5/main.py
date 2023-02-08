from collections.abc import Callable
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint, solve_ivp
from scipy.optimize import fsolve

from services.model import EconomicModel, G_by_x


t_start = 0
t_end = 500


def count_volume(model, data):
    legal = []
    shadow = []
    ratio = []
    for current_data in data:
        legal_and_shadow = model.Q1(current_data) + model.Q2(current_data)
        legal_c = model.Q1(current_data) / legal_and_shadow
        shadow_c = model.Q2(current_data) / legal_and_shadow
        ratio_c = model.Q2(current_data) / model.Q1(current_data)
        
        legal.append(legal_c)
        shadow.append(shadow_c)
        ratio.append(ratio_c)
        
    return legal, shadow, ratio


def count_profit(model, data):
    legal = []
    shadow = []
    country = []
    for current_data in data:
        all_profit = (model.G1(current_data) + model.G2(current_data) + model.G(current_data))
        legal_c = model.G1(current_data) / all_profit
        shadow_c = model.G2(current_data) / all_profit
        country_c = model.G(current_data) / all_profit
        
        legal.append(legal_c)
        shadow.append(shadow_c)
        country.append(country_c)

    return legal, shadow, country


def count_price(model, data):
    legal_p = data[:, 1]
    shadow_p = data[:, 5]
    legal_s = data[:, 2]
    shadow_s = data[:, 6]
    
    return legal_p, shadow_p, legal_s, shadow_s


def count_work_volume(model, data):
    legal = []
    shadow = []
    for current_data in data:
        legal_c = model.L1(current_data) / model.S1(current_data)
        shadow_c = model.L2(current_data) / model.S2(current_data)
        
        legal.append(legal_c)
        shadow.append(shadow_c)
    
    return legal, shadow


def count_fonds(model, data, init):
    legal = data[:, 3] / init[3]
    shadow = data[:, 7] / init[7]
    
    return legal, shadow


def f_by_tau_sigma(
    model: EconomicModel,
    target_func: Callable,
    tau: float, sigma: float,
    init: list[float] | None = None) -> float:
    prepared_model_call = lambda x, *args: model(x, changed_params={"tau": args[0], "sigma": args[1]})
    
    # начальные значения модели
    if init is None:
        init = [0, 0.5, 0.25, 0.1, 0, 0.5, 0.25, 0.1]
    
    new_x = fsolve(prepared_model_call, x0=init, args=(tau, sigma))
    
    # Проверка правильно ли работает решение с fsolve. Т.к. оно занимает намного меньше рессурсов для выполнения.
    # t = np.arange(t_start, t_end, 0.1)
    # new_x = odeint(lambda t, x, *args: prepared_model_call(x, *args), init, t, tfirst=True, args=(tau, sigma))[-1]
    
    model.tau = tau
    model.sigma = sigma
    result = target_func(new_x)
    
    # сброс до базовых значений модели
    model.set_default()
    return result

    
class ABSPlotFuncByTauAndSigma:
    def __init__(self):
        self.levels = 10
        
    def __call__(self):
        x, y, z = self.count()
        self.plot(x, y, z)
        return self.prepared_func
        
    def prepared_func(self, tau, sigma):
        return f_by_tau_sigma(self.model, self.target_func, tau, sigma)
        
    def count(self):
        z = []
        x = np.linspace(0, 1, 100)
        y = np.linspace(0, 1, 100)
        
        for sigma in y:
            z.append([])
            for tau in x:
                res = self.prepared_func(tau, sigma)
                z[-1].append(res)
                
        return x, y, z
        
    def plot(self, x, y, z):
        ax1 = plt.subplot(1, 1, 1)
        ax1.set_xlabel('tau')
        ax1.set_ylabel('sigma')
        
        ax1.contourf(x, y, z, levels=self.levels, cmap='plasma')
        cs = ax1.contour(x, y, z, levels=self.levels, colors=np.zeros((self.levels, 3)))
        ax1.clabel(cs)
        
        plt.show()


def main():
    line_width = 0.5
    t = np.arange(t_start, t_end, 0.1)
    model = EconomicModel()
    
    p_plot = ABSPlotFuncByTauAndSigma()
    
    # Вывод зависомости G от tau и sigma
    # p_plot.model = model
    # p_plot.target_func = model.G
    # p_plot()
    
    # Вывод зависомости G1 от tau и sigma
    # p_plot.model = model
    # p_plot.target_func = model.G1
    # p_plot()
    
    # Вывод зависомости G2 от tau и sigma
    # p_plot.model = model
    # p_plot.target_func = model.G2
    # p_plot()
    
    # Пример оптимизации системы
    # a, b, c = 0.6, 0.3, 0.1
    # p_plot.model = model
    # p_plot.levels = 30
    # p_plot.target_func = lambda x: (a * model.G(x)) + (b * model.G1(x)) - (c * model.G2(x))
    # p_plot()

    model.tau = 0.45
    model.sigma = 0
    # print(model.tau)
    # print(model.sigma)
    
    f = lambda t, x: model(x)
    
    init = [0, 0.5, 0.25, 0.1, 0, 0.5, 0.25, 0.1]
    result = odeint(f, init, t, tfirst=True)
    
    l_v, s_v, r_v = count_volume(model, result)
    l_p, s_p, r_p = count_profit(model, result)
    l_price, s_price, l_salary, s_salary = count_price(model, result)
    l_wv, s_wv = count_work_volume(model, result)
    l_f, s_f = count_fonds(model, result, init)
    
    ax1 = plt.subplot(2, 2, 1)
    ax2 = plt.subplot(2, 2, 2)
    ax3 = plt.subplot(2, 2, 3)
    ax4 = plt.subplot(2, 2, 4)
    
    plt.subplots_adjust(wspace=0.2, hspace=0.5)
    ax1.set_title('A')
    ax1.set_xlabel('t')
    ax1.set_ylabel('value')
    
    ax2.set_title('Б')
    ax2.set_xlabel('t')
    ax2.set_ylabel('value')
    
    ax3.set_title('В')
    ax3.set_xlabel('t')
    ax3.set_ylabel('value')
    
    ax4.set_title('Г')
    ax4.set_xlabel('t')
    ax4.set_ylabel('value')
    
    ax1.plot(t, l_v, label='легальний', linewidth=line_width)
    ax1.plot(t, s_v, label='тіньовий', linewidth=line_width)
    ax1.plot(t, r_v, label='коефіцієнт тінізації', linewidth=line_width)
    
    ax2.plot(t, l_p, label='легальний', linewidth=line_width)
    ax2.plot(t, s_p, label='тіньовий',linewidth=line_width)
    ax2.plot(t, r_p, label='держава',linewidth=line_width)
    
    ax3.plot(t, l_price, label='легальний (ціни)', linewidth=line_width)
    ax3.plot(t, s_price, label='тіньовий (ціни)', linewidth=line_width)
    ax3.plot(t, l_salary, label='легальний (ЗП)', linewidth=line_width)
    ax3.plot(t, s_salary, label='тіньовий (ЗП)', linewidth=line_width)
    
    ax4.plot(t, l_wv, label='легальний (трудові ресурси)', linewidth=line_width)
    ax4.plot(t, s_wv, label='тіньовий (трудові ресурси)', linewidth=line_width)
    ax4.plot(t, l_f, label='легальний (обсяг фондів)', linewidth=line_width)
    ax4.plot(t, s_f, label='тіньовий (обсяг фондів)', linewidth=line_width)
    
    ax1.legend()
    ax2.legend()
    ax3.legend()
    ax4.legend()
    
    plt.show()
    
    
if __name__ == '__main__':
    main()