import matplotlib.pyplot as plt
import numpy as np

from services.opt import *
from services.opt import GoldenSectionMethod
from services.model import EconomicModel, G_by_x


def task1():
    f = lambda x:  2 - (1 / (np.log2(x**4 + 4*(x**3) + 29)))
    x0 = -1
    eps = 0.01
    
    golden_section_method = GoldenSectionMethod()
    
    step = 0.001
    interval = golden_section_method.get_interval(f, x0, step)
    print(f'Sven method interval with {step=}: {interval}')
    golden_section_method.is_plot = True
    golden_section_method(f, interval, eps)

    
def task2():
    model = EconomicModel()
    profit = lambda x: G_by_x(model, x)
    
    golden_section_method = GoldenSectionMethod()
    golden_section_method.is_minimization = False
    golden_section_method.is_plot = True
    golden_section_method.is_log = True
    golden_section_method(profit, (0, 1), 0.001)
    
    
def main():
    task1()
    task2()
    
    
if __name__ == '__main__':
    main()