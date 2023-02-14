import matplotlib; matplotlib.use('Qt5Agg')
import numpy as np
import math
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout
import sip
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.backend_bases import MouseButton
import matplotlib.pyplot as plt
import sympy
from numpy import pi
from collections.abc import Callable
from scipy.integrate import odeint, solve_ivp
from scipy.optimize import fsolve


# import UI
from py_UI.app_ui import Ui_MainWindow

# from count_expression import count_expression
# from count_diff import count_function_diff, get_grad_func

from services.model import EconomicModel
from method import test_func, gradient_descent, get_rotated_matrix
from model import get_target_func



def plot_levels(fig, ax, params, data, levels=5, *, contour_type='line_label'):
    '''
    contour_type = 'line' | 'fill' | 'line_label' | 'fill_label'
    '''

    x, y = params[0]['value'], params[1]['value']

    # fig, ax = plt.subplots()

    match contour_type:
        case 'line':
            ax.contour(x, y, data, levels=levels, cmap='plasma')
        case 'line_label':
            cs = ax.contour(x, y, data, levels=levels, cmap='plasma')
            ax.clabel(cs)
        case 'fill':
            ax.contourf(x, y, data, levels=levels, cmap='plasma')
        case 'fill_label':
            color_line = np.zeros((levels, 3))
            c = np.linspace(1, 0.2, levels)
            # color_line[:, 0] = c
            # color_line[:, 1] = c
            # color_line[:, 2] = c
            color_line[:, 0] = 0
            color_line[:, 1] = 0
            color_line[:, 2] = 0
            ax.contourf(x, y, data, levels=levels, cmap='plasma')
            cs = ax.contour(x, y, data, levels=levels, colors=color_line)
            ax.clabel(cs)

    fig.set_figwidth(5)
    fig.set_figheight(5)


def plot_path(ax, points, label, color='r'):
    x, y = points[0], points[1]
    ax.plot(x, y, color=color, label=label)
    
    
    

def f_by_tau_sigma(
    model: EconomicModel,
    target_func: Callable,
    tau: float, sigma: float,
    init: list[float] | None = None) -> float:
    prepared_model_call = lambda x, *args: model(x, changed_params={"tau": args[0], "sigma": args[1]})
    # print(tau, sigma)
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


    

# A = [[2, 1], [1, 2]]
# b = [0, 0]
# target_func = lambda p: test_func(*p, A, b)


# model = EconomicModel()
# target_func = lambda p: f_by_tau_sigma(model, model.G, *p)


# es = [1, 10, 100]
# e = es[2]
# angle = math.pi/2
# A = [[1, 0], [0, e]]
# b = [0, 0]
# func = get_rotated_matrix(angle)(test_func)
# target_func = lambda p: func(*p, A, b)

# 0 16
func = get_target_func()
target_func = lambda p: func(*p) 


class GraphicWidget(QWidget):
    def __init__(self, parent=None):
        QWidget.__init__(self, parent)
        
        self.plot_layout = QVBoxLayout(self)
        
        self._figure = Figure(facecolor='#20242d')
        self._canvas = FigureCanvas(self._figure)
        self._axis = self._figure.add_subplot(111)
        self._init_axis()
        self._canvas.draw()
        self.toolbar = NavigationToolbar(self._canvas, self)
        
        self.plot_layout.addWidget(self._canvas)
        self.plot_layout.addWidget(self.toolbar)
        
        self.levels = 10
        self.contour_type = 'line'
        self.max_count_init = [0, 0]
        
        self.is_legend = True
        self.number_of_max_points = 0
        
        self._canvas.mpl_connect('button_press_event', self.mpl_on_click)
        
        self.func = target_func
        # self.is_max = False
        self.is_max = True
        
    def mpl_on_click(self, event):
        if event.dblclick:
            x, y = event.xdata, event.ydata
            if event.button is MouseButton.RIGHT:
                x, y = 1, 1
            points = self.count_max(x, y)
            self.plot_path(points)
        
    def _init_axis(self):
        color = '#ffffff'
        
        self._axis.tick_params(axis='x', colors=color)
        self._axis.tick_params(axis='y', colors=color)
        
        self._axis.xaxis.label.set_color(color)
        self._axis.yaxis.label.set_color(color)
        
    def _clear_plot(self):
        self.number_of_max_points = 0
        self._axis.clear()
        
    def update_plot(self):
        self._clear_plot()
        self._axis.set_xlabel('X')
        self._axis.set_ylabel('Y')
            
        params = self._get_params()
        data = self._count_data()
        plot_levels(self._figure, self._axis, params, data, self.levels, contour_type=self.contour_type)
        
        self._canvas.draw()
        self._canvas.flush_events()
        
    def set_grid(self, grid):
        self.x, self.y = grid
        
    def set_exp(self, expression):
        ...
        # self.expression = expression
        
    def plot_path(self, points):
        self._axis.plot(
            points[0], points[1],
            color="blue",
            zorder=1,
            label=f'iterations ({self.number_of_max_points}): {points.shape[1]}'
        )
        self._axis.scatter(
            points[0], points[1],
            zorder=2,
            color="blue",
        )
        self._axis.scatter(
            points[0][0], points[1][0],
            color="#00ff00",
            zorder=2,
            label=f"start [x={points[0][0]:.5f}, y={points[1][0]:.5f}] ({self.number_of_max_points})"
        )
        self._axis.scatter(
            points[0][-1], points[1][-1],
            color="#ff0000",
            zorder=2,
            label=f"extremum [x={points[0][-1]:.5f}, y={points[1][-1]:.5f}] ({self.number_of_max_points})"
        )
        
        if self.is_legend:
            self._axis.legend()
            
        self._canvas.draw()
        self._canvas.flush_events()
        
        self.number_of_max_points += 1
        
    def _count_max(self):
        eps = 0.001
        bounds = [[self.x[0][0], self.x[-1][0]], [self.y[0][0], self.y[0][-1]]]

        grad_iters = []
        func = self.func
        if self.is_max == True:
            func = lambda *args, **kwargs: -self.func(*args, **kwargs)
        
        x, y = gradient_descent(func, self.max_count_init, eps, bounds=bounds, iterations=grad_iters)
        print(f'Extremum: x={x} y={y} f(x)={self.func([x, y])}')
        
        grad_iters = np.array(grad_iters)
        return np.array([grad_iters[:, 0], grad_iters[:, 1]])
            
    def count_max(self, x, y):
        self.max_count_init = [x, y]
        return self._count_max()
        
    def _count_data(self):
        data = []
        for x in self.x[:, 0]:
            data.append([])
            for y in self.y[0]:
                data[-1].append(self.func([x, y]))
        
        return np.array(data)
        
    def _get_params(self):
        return {
            0: {
                'name': 'x',
                'value': self.x,
            },
            1: {
                'name': 'y',
                'value': self.y,
            }
        }
    
    
class MyUi_MainWindow(Ui_MainWindow):
    def setupUi(self, MainWindow, *args, **kwargs):
        super().setupUi(MainWindow, *args, **kwargs)
        
        self.graphic = GraphicWidget(MainWindow)
        self.main_layout.insertWidget(0, self.graphic)
        self.graphic.show()

        self._init_defaults()
        self._init_events()
        
        
    def _count_grid(self):
        xmin = float(self.x_min_edit.text())
        xmax = float(self.x_max_edit.text())
        
        ymin = float(self.y_min_edit.text())
        ymax = float(self.y_max_edit.text())
        
        return np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    
    def _set_grid(self):
        self.graphic.set_grid(self._count_grid())
        
    def _set_levels(self):
        self.graphic.levels = self.levels_input.value()
        
    def _set_exp(self):
        self.graphic.set_exp(self.exp_input.text())
        
    def _set_draw_type(self):
        self.graphic.contour_type = self.draw_type_input.currentText()
        
    def _init_events(self):
        self.draw_btn.clicked.connect(self.graphic.update_plot)
        
        self.x_min_edit.editingFinished.connect(self._set_grid)
        self.x_max_edit.editingFinished.connect(self._set_grid)
        self.y_min_edit.editingFinished.connect(self._set_grid)
        self.y_max_edit.editingFinished.connect(self._set_grid)
        
        self.levels_input.editingFinished.connect(self._set_levels)
        
        self.exp_input.editingFinished.connect(self._set_exp)
        
        self.draw_type_input.currentTextChanged.connect(self._set_draw_type)
        
    def _init_defaults(self):
        self.x_min_edit.setText(str(-6))
        self.x_max_edit.setText(str(6))
        self.y_min_edit.setText(str(-6))
        self.y_max_edit.setText(str(6))
        self._set_grid()
        
        self.levels_input.setValue(10)
        self._set_levels()
        
        # test_exp = '-1.1*(x**2) - 1.5*(y**2) + 2*x*y + x + 5'
        # self.exp_input.setText(test_exp)
        self.exp_input.setEnabled(False)
        # self._set_exp()
        
        self._set_draw_type()


def main():
    import sys
    app = QApplication(sys.argv)

    mainWindow = QMainWindow()

    ui = MyUi_MainWindow()
    ui.setupUi(mainWindow)

    mainWindow.show()

    app.exec()


if __name__ == '__main__':
    main()
