import matplotlib; matplotlib.use('Qt5Agg')
import numpy as np
import math
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout
import sip
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import sympy
from numpy import pi


# import UI
from py_UI.app_ui import Ui_MainWindow

from count_expression import count_expression

def input_expression(help_str='Input expression: '):
    data_str = input(help_str)
    return data_str


def input_params():
    from numpy import pi
    xmin, xmax = map(eval, input('Input xmin, xmax: ').split(', '))
    ymin, ymax = map(eval, input('Input ymin, ymax: ').split(', '))
    x, y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    params = {}
    params[0] = {}
    params[0]['name'] = 'x'
    params[0]['value'] = x
    params[1] = {}
    params[1]['name'] = 'y'
    params[1]['value'] = y
    return params


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


def find_extremum(grad_fun, init, n_iterations, eta, b, *, eps=1e-6):
    points = [np.array(init)]
    v = 0
    for k in range(n_iterations):
        x = points[-1]
        x = x + v * b - eta * grad_fun(x[0] + b * v, x[1] + b * v)
        
        dist = math.hypot(points[-1][0] - x[0], points[-1][1] - x[1])
        points.append(x)
        if dist < eps:
            break
        
    return np.array(points).T


def plot_path(ax, points, label, color='r'):
    x, y = points[0], points[1]
    ax.plot(x, y, color=color, label=label)
    
    
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
        
    def mpl_on_click(self, event):
        if event.dblclick:
            x, y = event.xdata, event.ydata
            self.count_max(x, y)
        
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
        self.expression = expression
        
    def _count_max(self):
        x, y = sympy.symbols('x y')
        exp = eval(self.expression)
        x_d = str(sympy.diff(exp, x))
        y_d = str(sympy.diff(exp, y))
        
        grad = lambda x, y: np.array([-eval(x_d), -eval(y_d)])
        
        points = find_extremum(grad, self.max_count_init, n_iterations=10000, eta=0.01, b = 0.5, eps=1e-6)
        
        self._axis.plot(
            points[0], points[1],
            color="blue",
            label=f'iterations ({self.number_of_max_points}): {points.shape[1]}'
        )
        self._axis.plot(
            points[0][0], points[1][0],
            marker="o",
            markersize=5,
            markerfacecolor="green",
            label=f"start [x={points[0][0]:.5f}, y={points[1][0]:.5f}] ({self.number_of_max_points})"
        )
        self._axis.plot(
            points[0][-1], points[1][-1],
            marker="o",
            markersize=5,
            markerfacecolor="red",
            label=f"maximum [x={points[0][-1]:.5f}, y={points[1][-1]:.5f}] ({self.number_of_max_points})"
        )
        
        if self.is_legend:
            self._axis.legend()
            
        self._canvas.draw()
        self._canvas.flush_events()
        
        self.number_of_max_points += 1
            
    def count_max(self, x, y):
        self.max_count_init = [x, y]
        self._count_max()
        
    def _count_data(self):
        return count_expression(self._get_params(), self.expression)
        
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
        self.x_min_edit.setText(str(0))
        self.x_max_edit.setText(str(2))
        self.y_min_edit.setText(str(0))
        self.y_max_edit.setText(str(2))
        self._set_grid()
        
        self.levels_input.setValue(10)
        self._set_levels()
        
        test_exp = '-1.1*(x**2) - 1.5*(y**2) + 2*x*y + x + 5'
        self.exp_input.setText(test_exp)
        self._set_exp()
        
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
