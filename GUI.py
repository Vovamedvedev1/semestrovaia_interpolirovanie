import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

class GUI:
    def __init__(self, root, fig):
        self.root = root
        self.root.title("Интерполяция сплайнами")
        self.root.geometry("1200x1000")

        
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=0)  
        root.rowconfigure(1, weight=1)  

        
        input_frame = ttk.Frame(root, padding="10")
        input_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)

        graph_frame = ttk.Frame(root)
        graph_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)

        font_large = ('Arial', 14)  
        font_medium = ('Arial', 13)  

        ttk.Label(input_frame, text="Функция f(x):", font=font_large).grid(column=0, row=0, sticky=tk.W, padx=(0, 5))
        self.entry_func = ttk.Entry(input_frame, width=15, font=font_medium)
        self.entry_func.grid(column=1, row=0, sticky=(tk.W, tk.E), padx=(0, 20))
        self.entry_func.insert(0, "x**2")

        ttk.Label(input_frame, text="Левая граница (a):", font=font_large).grid(column=2, row=0, sticky=tk.W, padx=(0, 5))
        self.entry_a = ttk.Entry(input_frame, width=10, font=font_medium)
        self.entry_a.grid(column=3, row=0, sticky=(tk.W, tk.E), padx=(0, 20))
        self.entry_a.insert(0, "0.0")

        ttk.Label(input_frame, text="Правая граница (b):", font=font_large).grid(column=4, row=0, sticky=tk.W, padx=(0, 5))
        self.entry_b = ttk.Entry(input_frame, width=10, font=font_medium)
        self.entry_b.grid(column=5, row=0, sticky=(tk.W, tk.E), padx=(0, 20))
        self.entry_b.insert(0, "2.0")

        ttk.Label(input_frame, text="Количество узлов (n):", font=font_large).grid(column=6, row=0, sticky=tk.W, padx=(0, 5))
        self.entry_n = ttk.Entry(input_frame, width=10, font=font_medium)
        self.entry_n.grid(column=7, row=0, sticky=(tk.W, tk.E), padx=(0, 20))
        self.entry_n.insert(0, "5")

        self.calculate_btn = ttk.Button(
            input_frame,
            text="Рассчитать",
            command=self.calculate,
            style="Large.TButton"
        )
        self.calculate_btn.grid(column=8, row=0, padx=(10, 5), sticky=tk.EW)

        self.scipy_btn = ttk.Button(
            input_frame,
            text="SciPy Сплайн",
            command=self.get_scipy_spline_interpolation,
            style="Large.TButton"
        )
        self.scipy_btn.grid(column=9, row=0, padx=(5, 10), sticky=tk.EW)

        self.error_label = ttk.Label(input_frame, text="", foreground="red",
                                    font=('Arial', 11), wraplength=300, justify=tk.LEFT)
        self.error_label.grid(column=10, row=0, padx=(20, 0), sticky=tk.W)

        for i in range(11):  # Увеличиваем до 11 колонок
            input_frame.columnconfigure(i, weight=1 if i < 10 else 2)
        input_frame.rowconfigure(0, weight=1)

        self.fig = fig
        self.fig.set_size_inches(14, 9)
        self.canvas = FigureCanvasTkAgg(self.fig, master=graph_frame)
        self.canvas.draw()
        self.toolbar = NavigationToolbar2Tk(self.canvas, graph_frame)
        self.toolbar.update()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        graph_frame.columnconfigure(0, weight=1)
        graph_frame.rowconfigure(0, weight=1)

        style = ttk.Style()
        style.configure("Large.TButton", font=('Arial', 13), padding=(10, 8))

        root.minsize(1200, 800)

    def calculate(self):
        pass

    def get_scipy_spline_interpolation(self):
        pass
'''def get_chebushev_nodes(self):
        k = np.arange(1, self.n + 1)
        cheb_nodes = np.cos((2 * k - 1) * np.pi / (2 * self.n))
        return 0.5 * (self.a + self.b) + 0.5 * (self.b - self.a) * cheb_nodes'''


'''def get_gauss_polinom(self, t):
        diff_table = np.zeros((self.n, self.n))
        for i in range(self.n):
            diff_table[i, 0] = self.y[i]
        for j in range(1, self.n):
            for i in range(self.n - j):
                diff_table[i, j] = (diff_table[i + 1, j - 1] - diff_table[i, j - 1]) / (self.x[i + j] - self.x[i])
        result = np.zeros_like(t)
        for k in range(self.n):
            tt = diff_table[0, k]
            for j in range(k):
                tt *= (t - self.x[j])
            result += tt
        return result'''