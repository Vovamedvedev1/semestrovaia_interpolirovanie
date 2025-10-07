import functools
import numpy as np
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from GUI import GUI
import matplotlib.pyplot as plt
import math
from math import factorial
import sympy as sp
from sympy.utilities.lambdify import lambdify
from sympy.parsing.sympy_parser import parse_expr, function_exponentiation
from scipy.interpolate import interp1d, CubicSpline

 
def function(x, math_value):
    compiled_value = compile(math_value, '<string>', 'eval')
    return eval(compiled_value, {'np': np, 'math': math}, {'x': x})
 
class InterpolationApp(GUI):
    def __init__(self, root):
        self.x_plot, self.y_plot, self.y_lagrange = None, None, None
        self.x, self.y = None, None
        self.a, self.b, self.n = None, None, None
        self.function_string = None
        self.fig, self.axes = plt.subplots(2, 2, figsize=(15, 10))
        super().__init__(root, self.fig)
 
    def get_chebushev_nodes(self):
        k = np.arange(1, self.n + 1)
        cheb_nodes = np.cos((2 * k - 1) * np.pi / (2 * self.n))
        return 0.5 * (self.a + self.b) + 0.5 * (self.b - self.a) * cheb_nodes
 
    def get_linear_spline(self, t):
        result = np.zeros_like(t)
        for i in range(len(t)):
            k = 0
            while k < len(self.x) - 1 and t[i] > self.x[k + 1]:
                k += 1
            x0, x1 = self.x[k], self.x[k + 1]
            y0, y1 = self.y[k], self.y[k + 1]
            result[i] = y0 + (y1 - y0) * (t[i] - x0) / (x1 - x0)
        return result
    
    def get_parameters_parabol_spline(self):
        a, b, c = np.zeros(self.n - 1), np.zeros(self.n - 1), np.zeros(self.n - 1)
        x0, x1, x2 = self.x[0], self.x[1], self.x[2]
        y0, y1, y2 = self.y[0], self.y[1], self.y[2]
        A = ((y2 - y0)*(x1 - x0) - (y1 - y0)*(x2 - x0)) / ((x2**2 - x0**2)*(x1 - x0) - (x1**2 - x0**2)*(x2 - x0))
        B = (y1 - y0 - A*(x1**2 - x0**2)) / (x1 - x0)
        b[0] = 2*A*x0 + B
        for i in range(0, self.n - 1):
            h = self.x[i + 1] - self.x[i]
            a[i] = self.y[i]
            if i < self.n - 2:
                b[i + 1] = -b[i] + 2 * (self.y[i + 1] - self.y[i]) / h   
            c[i] = (self.y[i + 1] - a[i] - b[i] * h) / (h ** 2)
        return a, b, c
    
    def get_parabol_spline(self, t):
        result = np.zeros_like(t) 
        for i in range(len(t)):
            k = 0
            while k < len(self.x) - 1 and t[i] > self.x[k + 1]:
                k += 1
            xk = self.x[k]
            result[i] = self.a_parabol[k] + self.b_parabol[k] * (t[i] - xk)  + self.c_parabol[k] * (t[i] - xk) ** 2
        return result 
    
    def get_parameters_cubic_spline(self):
        A, B, D, F = np.zeros(self.n - 2), np.zeros(self.n - 2), np.zeros(self.n - 2), np.zeros(self.n - 2)
        a, b, c, d = np.zeros(self.n - 1), np.zeros(self.n - 1), np.zeros(self.n), np.zeros(self.n - 1) 
        alpha, betta, h = np.zeros(self.n - 1), np.zeros(self.n - 1), np.zeros(self.n - 1)
        h[0], alpha[0], betta[0] = self.x[1] - self.x[0], 0, 0
        for i in range(0, self.n - 2, 1):
            h[i+1] = self.x[i + 2] - self.x[i + 1]
            A[i], B[i], D[i] = h[i + 1], 2 * (h[i] + h[i + 1]), h[i]
            F[i] = 6 * ((self.y[i + 2] - self.y[i + 1]) / h[i + 1] - (self.y[i + 1] - self.y[i]) / h[i])
            alpha[i + 1] = -A[i] / (B[i] + alpha[i] * D[i])
            betta[i + 1] = (F[i] - D[i] * betta[i]) / (B[i] + alpha[i] * D[i])
        c[self.n - 1] = 0
        for i in range(self.n - 2, -1, -1):
            c[i] = 0 if i == 0 else alpha[i] * c[i + 1] + betta[i]
            a[i] = self.y[i + 1]
            d[i] = (c[i + 1] - c[i]) / h[i]
            b[i] = c[i + 1] * h[i] / 2  - (d[i] * h[i] ** 2) / 6 + (self.y[i + 1] - self.y[i]) / h[i]
        return a, b, c, d
    
    def get_cubic_spline(self, t):
        result = np.zeros_like(t) 
        for i in range(len(t)):
            k = 0
            while k < len(self.x) - 1 and t[i] > self.x[k + 1]:
                k += 1
            k += 1
            xk = self.x[k]
            result[i] = self.a_cubic[k - 1] + self.b_cubic[k - 1] * (t[i] - xk)  + (self.c_cubic[k] * (t[i] - xk) ** 2) / 2 + (self.d_cubic[k - 1] * (t[i] - xk) ** 3) / 6
        return result

    def calculate(self):
        try:
            self.function_string = str(self.entry_func.get())
            self.a = float(self.entry_a.get())
            self.b = float(self.entry_b.get())
            self.n = int(self.entry_n.get())
            if self.n <= 3:
                raise ValueError("Количество узлов должно быть положительным и большим 3")
            if self.a >= self.b:
                raise ValueError("Левая граница должна быть меньше правой")
            self.x_plot = np.linspace(self.a, self.b, 100000)
            self.y_plot = function(self.x_plot, math_value=self.function_string)
            self.x = np.linspace(self.a, self.b, self.n)
            self.y = function(self.x, math_value=self.function_string)
            self.linear_spline = self.get_linear_spline(self.x_plot)
            self.a_cubic, self.b_cubic, self.c_cubic, self.d_cubic = self.get_parameters_cubic_spline()
            self.cubic_spline = self.get_cubic_spline(self.x_plot)
            self.a_parabol, self.b_parabol, self.c_parabol = self.get_parameters_parabol_spline()
            self.parabol_spline = self.get_parabol_spline(self.x_plot)
            actual_error_linear = np.abs(self.y_plot - self.linear_spline)
            actual_error_cubic = np.abs(self.y_plot - self.cubic_spline)
            actual_error_parabol = np.abs(self.y_plot - self.parabol_spline)
            self.axes[0][0].clear()
            self.axes[0][1].clear()
            self.axes[1][0].clear()   
            self.axes[1][1].clear()   
 
            self.axes[0][0].plot(self.x_plot, self.y_plot, label='Исходная функция', linewidth=2)
            self.axes[0][0].plot(self.x_plot, self.linear_spline, 'g--', label='Линейный сплайн', linewidth=1.5)
            self.axes[0][0].scatter(self.x, self.y, c='r', s=50, label='Узлы интерполяции', zorder=5)
            self.axes[0][0].set_xlabel('x')
            self.axes[0][0].set_ylabel('y')
            self.axes[0][0].set_title(f'Интерполяция линейным сплайном (n={self.n})')
            self.axes[0][0].legend()
            self.axes[0][0].grid(True, alpha=0.3)

            self.axes[0][1].plot(self.x_plot, self.y_plot, label='Исходная функция', linewidth=2)
            self.axes[0][1].plot(self.x_plot, self.cubic_spline, 'g--', label='Кубический сплайн', linewidth=1.5)
            self.axes[0][1].scatter(self.x, self.y, c='r', s=50, label='Узлы интерполяции', zorder=5)
            self.axes[0][1].set_xlabel('x')
            self.axes[0][1].set_ylabel('y')
            self.axes[0][1].set_title(f'Интерполяция кубическим сплайном (n={self.n})')
            self.axes[0][1].legend()
            self.axes[0][1].grid(True, alpha=0.3)

            self.axes[1][0].plot(self.x_plot, self.y_plot, label='Исходная функция', linewidth=2)
            self.axes[1][0].plot(self.x_plot, self.parabol_spline, 'g--', label='Параболический сплайн', linewidth=1.5)
            self.axes[1][0].scatter(self.x, self.y, c='r', s=50, label='Узлы интерполяции', zorder=5)
            self.axes[1][0].set_xlabel('x')
            self.axes[1][0].set_ylabel('y')
            self.axes[1][0].set_title(f'Интерполяция параболическим сплайном (n={self.n})')
            self.axes[1][0].legend()
            self.axes[1][0].grid(True, alpha=0.3)


            self.axes[1][1].plot(self.x_plot, actual_error_linear, label='Погрешность линеного сплайна', linewidth=2)
            self.axes[1][1].plot(self.x_plot, actual_error_parabol, label='Погрешность параболического сплайна', linewidth=2)
            self.axes[1][1].plot(self.x_plot, actual_error_cubic, label='Погрешность кубического сплайна', linewidth=2) 
            self.axes[1][1].set_xlabel('x')
            self.axes[1][1].set_ylabel('Фактическая погрешность')
            self.axes[1][1].set_title(f'Максимальное значение фактической погрешности:\nЛинейный: {actual_error_linear.max()}; Параболический: {actual_error_parabol.max()}; Кубический: {actual_error_cubic.max()};')
            self.axes[1][1].legend()
            self.axes[1][1].grid(True, alpha=0.3)
 
            self.fig.tight_layout()
            self.canvas.draw()
            self.error_label.config(text="")
        except Exception as e:
            self.error_label.config(text=f"Ошибка: {str(e)}")

    def get_scipy_spline_interpolation(self):
        try:
            scipy_window = tk.Toplevel(self.root)
            scipy_window.title("SciPy Spline Interpolation")
            scipy_window.geometry("1200x800")
            frame = ttk.Frame(scipy_window)
            frame.pack(fill=tk.BOTH, expand=True)
            fig_scipy = Figure(figsize=(12, 8))
            axes_scipy = fig_scipy.subplots(2, 2)
            ls = interp1d(self.x, self.y, kind='linear')
            linear_spline_scipy = ls(self.x_plot)
            cs = CubicSpline(self.x, self.y, bc_type='natural')
            cubic_spline_scipy = cs(self.x_plot)
            ps = interp1d(self.x, self.y, kind='quadratic')
            parabol_spline_scipy = ps(self.x_plot)
            ls_error = np.abs(self.y_plot - linear_spline_scipy)
            ps_error = np.abs(self.y_plot - parabol_spline_scipy)
            cs_error = np.abs(self.y_plot - cubic_spline_scipy)
            axes_scipy[0, 0].plot(self.x_plot, self.y_plot, label='Исходная функция', linewidth=2)
            axes_scipy[0, 0].plot(self.x_plot, linear_spline_scipy, 'g--', label='Линейный сплайн (SciPy)', linewidth=1.5)
            axes_scipy[0, 0].scatter(self.x, self.y, c='r', s=50, label='Узлы интерполяции', zorder=5)
            axes_scipy[0, 0].set_xlabel('x')
            axes_scipy[0, 0].set_ylabel('y')
            axes_scipy[0, 0].set_title(f'Интерполяция линейным сплайном (n={self.n}) - SciPy')
            axes_scipy[0, 0].legend()
            axes_scipy[0, 0].grid(True, alpha=0.3)
            axes_scipy[0, 1].plot(self.x_plot, self.y_plot, label='Исходная функция', linewidth=2)
            axes_scipy[0, 1].plot(self.x_plot, cubic_spline_scipy, 'g--', label='Кубический сплайн (SciPy)', linewidth=1.5)
            axes_scipy[0, 1].scatter(self.x, self.y, c='r', s=50, label='Узлы интерполяции', zorder=5)
            axes_scipy[0, 1].set_xlabel('x')
            axes_scipy[0, 1].set_ylabel('y')
            axes_scipy[0, 1].set_title(f'Интерполяция кубическим сплайном (n={self.n}) - SciPy')
            axes_scipy[0, 1].legend()
            axes_scipy[0, 1].grid(True, alpha=0.3)
            axes_scipy[1, 0].plot(self.x_plot, self.y_plot, label='Исходная функция', linewidth=2)
            axes_scipy[1, 0].plot(self.x_plot, parabol_spline_scipy, 'g--', label='Параболический сплайн (SciPy)', linewidth=1.5)
            axes_scipy[1, 0].scatter(self.x, self.y, c='r', s=50, label='Узлы интерполяции', zorder=5)
            axes_scipy[1, 0].set_xlabel('x')
            axes_scipy[1, 0].set_ylabel('y')
            axes_scipy[1, 0].set_title(f'Интерполяция параболическим сплайном (n={self.n}) - SciPy')
            axes_scipy[1, 0].legend()
            axes_scipy[1, 0].grid(True, alpha=0.3)
            axes_scipy[1, 1].plot(self.x_plot, ls_error, label='Погрешность линейного сплайна', linewidth=2)
            axes_scipy[1, 1].plot(self.x_plot, ps_error, label='Погрешность параболического сплайна', linewidth=2)
            axes_scipy[1, 1].plot(self.x_plot, cs_error, label='Погрешность кубического сплайна', linewidth=2)
            axes_scipy[1, 1].set_xlabel('x')
            axes_scipy[1, 1].set_ylabel('Фактическая погрешность')
            axes_scipy[1, 1].set_title(f'Максимальная погрешность (SciPy):\nЛинейный: {ls_error.max()}; Параболический: {ps_error.max()}; Кубический: {cs_error.max()};')
            axes_scipy[1, 1].legend()
            axes_scipy[1, 1].grid(True, alpha=0.3)
            fig_scipy.tight_layout()
            canvas = FigureCanvasTkAgg(fig_scipy, master=frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            close_button = ttk.Button(scipy_window, text="Закрыть", 
                                    command=scipy_window.destroy)
            close_button.pack(pady=10)
            self.error_label.config(text="")
        except Exception as e:
            self.error_label.config(text=f"Ошибка в SciPy интерполяции: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = InterpolationApp(root)
    root.mainloop()