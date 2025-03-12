import numpy as np  
import matplotlib.pyplot as plt
from control import tf, feedback, step_response, series, margin
import control
import warnings
import openai
import tkinter as tk
from tkinter import ttk, scrolledtext, simpledialog, messagebox, filedialog
import datetime
import os
import random
import glob
import json
from scipy.optimize import differential_evolution
import io
from PIL import Image, ImageTk  # For logo and formula rendering
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.figure as mplfig
import base64
import threading

# For PDF report generation using ReportLab (pip install reportlab)
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# Suppress warning messages.
warnings.filterwarnings("ignore")

# ==============================
# OpenAI Settings for NLP
# ==============================

import openai
from dotenv import load_dotenv
import os
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
openai.default_headers = {"x-foo": "true"}


# ==============================
# Tooltip Class
# ==============================
class ToolTip:
    """A simple tooltip class."""
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tipwindow = None
        widget.bind("<Enter>", self.show_tip)
        widget.bind("<Leave>", self.hide_tip)
    def show_tip(self, event=None):
        if self.tipwindow or not self.text:
            return
        x, y, cx, cy = self.widget.bbox("insert")
        x = x + self.widget.winfo_rootx() + 25
        y = y + self.widget.winfo_rooty() + 20
        self.tipwindow = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(tw, text=self.text, justify='left',
                         background="#ffffe0", relief='solid', borderwidth=1,
                         font=("Tahoma", 10, "normal"))
        label.pack(ipadx=1)
    def hide_tip(self, event=None):
        if self.tipwindow:
            self.tipwindow.destroy()
            self.tipwindow = None

# ==============================
# Helper Function: Convert Transfer Function to LaTeX Format
# ==============================
def tf_to_latex(num, den):
    def poly_to_string(coeffs):
        terms = []
        degree = len(coeffs) - 1
        for i, coef in enumerate(coeffs):
            power = degree - i
            if abs(coef) < 1e-12:
                continue
            if power == 0:
                term = f"{coef:g}"
            elif power == 1:
                term = f"{coef:g}s" if abs(coef-1) >= 1e-12 else "s"
            else:
                term = f"{coef:g}s^{power}" if abs(coef-1) >= 1e-12 else f"s^{power}"
            terms.append(term)
        return " + ".join(terms) if terms else "0"
    num_str = poly_to_string(num)
    den_str = poly_to_string(den)
    return r"$\frac{" + num_str + "}{" + den_str + "}$"

# ==============================
# Helper Function: Render Formula Image
# ==============================
def render_formula_image(latex_formula):
    try:
        fig = plt.figure(figsize=(0.01, 0.01))
        text_obj = fig.text(0, 0, latex_formula, fontsize=20)
        fig.canvas.draw()
        bbox = text_obj.get_window_extent()
        width, height = bbox.width / fig.dpi, bbox.height / fig.dpi
        plt.close(fig)
        fig = plt.figure(figsize=(width, height), dpi=100)
        fig.text(0, 0, latex_formula, fontsize=20)
        plt.axis('off')
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.05, transparent=True)
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf)
        photo = ImageTk.PhotoImage(img)
        return photo
    except Exception as e:
        print("Formula render error:", e)
        return None

# ==============================
# Functions: Extract Target Values from Problem Description
# ==============================
def extract_target_settling_time(description):
    prompt = (
        "Extract the target settling time in seconds from the following control problem description. "
        "If no target is specified or the value is unclear, output 0.3. Only output the number.\n\n"
        f"Description: \"{description}\""
    )
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "system", "content": "Extract a number from the text and answer in English."},
                      {"role": "user", "content": prompt}]
        )
        result = response.choices[0].message.content.strip()
        try:
            return float(result)
        except:
            return 0.3
    except Exception as e:
        print("Error extracting settling time:", e)
        return 0.3

def extract_target_overshoot(description):
    prompt = (
        "Extract the target maximum overshoot percentage from the following control problem description. "
        "If not specified, output 5. Only output the number.\n\n"
        f"Description: \"{description}\""
    )
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "system", "content": "Extract a number from the text and answer in English."},
                      {"role": "user", "content": prompt}]
        )
        result = response.choices[0].message.content.strip()
        try:
            return float(result)
        except:
            return 5.0
    except Exception as e:
        print("Error extracting overshoot:", e)
        return 5.0

def extract_target_rise_time(description):
    prompt = (
        "Extract the target minimum rise time in seconds from the following control problem description. "
        "If not specified, output 0.5. Only output the number.\n\n"
        f"Description: \"{description}\""
    )
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "system", "content": "Extract a number from the text and answer in English."},
                      {"role": "user", "content": prompt}]
        )
        result = response.choices[0].message.content.strip()
        try:
            return float(result)
        except:
            return 0.5
    except Exception as e:
        print("Error extracting rise time:", e)
        return 0.5

# ==============================
# Sample Transfer Functions (20 samples, original versions)
# ==============================
sample_transfer_functions = [
    {"name": "Default Transfer Function", "num": [1], "den": [1,2,1], "desc": "Default system: tf([1], [1,2,1])"},
    {"name": "Example 1: Second-Order System", "num": [1], "den": [1,2,1], "desc": "Standard second-order system."},
    {"name": "Example 2: First-Order System", "num": [1], "den": [1,1], "desc": "First-order system."},
    {"name": "Example 3: Delayed System", "num": [1], "den": [1,3,2], "desc": "System with delay dynamics."},
    {"name": "Example 4: Low Damping System", "num": [1], "den": [1,0.5,1], "desc": "Second-order system with low damping."},
    {"name": "Example 5: High-Speed System", "num": [10], "den": [1,10], "desc": "Fast responding system."},
    {"name": "Example 6: Slow System", "num": [0.5], "den": [1,0.1], "desc": "Slow dynamic system."},
    {"name": "Example 7: Second Order with Low Damping", "num": [1], "den": [1,0.2,1], "desc": "Low damping system."},
    {"name": "Example 8: Second Order with High Damping", "num": [1], "den": [1,5,1], "desc": "High damping system."},
    {"name": "Example 9: Third-Order System", "num": [1], "den": [1,3,3,1], "desc": "Third-order system."},
    {"name": "Example 10: Simple Integrator", "num": [1], "den": [1,0], "desc": "Integrator system."},
    {"name": "Example 11: Simple Differentiator", "num": [1,0], "den": [1], "desc": "Differentiator system."},
    {"name": "Example 12: PID Example", "num": [0.01,1,0.5], "den": [1,0], "desc": "PID control structure (example values)."},
    {"name": "Example 13: Triple Polynomial", "num": [1], "den": [1,1,1], "desc": "Simple triple polynomial system."},
    {"name": "Example 14: Special Second-Order", "num": [2], "den": [1,4,4,1], "desc": "Special form second-order system."},
    {"name": "Example 15: Scaled System", "num": [1], "den": [1,2.5,1.5], "desc": "System with varied coefficients."},
    {"name": "Example 16: Third Order Polynomial", "num": [1], "den": [1,6,11,6], "desc": "Typical third-order system."},
    {"name": "Example 17: Minimal Damping", "num": [1], "den": [1,0.8,0.16], "desc": "System with very low damping."},
    {"name": "Example 18: Fourth-Order System", "num": [1], "den": [1,3,3,1,0.5], "desc": "A fourth-order system."},
    {"name": "Example 19: Simple Multi-Order System", "num": [0.5], "den": [1,2,2,1], "desc": "Balanced multi-order system."},
    {"name": "Example 20: Flat System", "num": [1], "den": [1,1,1,1], "desc": "Simple flat system."}
]

# ==============================
# PSO Optimization
# ==============================
def pso_optimize_pid(plant, target_settling_time, target_overshoot, target_rise_time, initial_pid,
                     swarm_size=20, max_iter=30):
    lb = np.array([2 * initial_pid['Kp'], 2 * initial_pid['Ki'], 1 * initial_pid['Kd']])
    ub = np.array([300 * initial_pid['Kp'], 200 * initial_pid['Ki'], 100 * initial_pid['Kd']])
    dim = 3
    w = 0.7
    c1 = 1.5
    c2 = 1.5
    swarm_positions = np.random.uniform(lb, ub, (swarm_size, dim))
    swarm_velocities = np.random.uniform(-1, 1, (swarm_size, dim))
    pbest_positions = swarm_positions.copy()
    pbest_costs = np.full(swarm_size, np.inf)
    gbest_position = None
    gbest_cost = np.inf
    ph_agent = PyHesapAgent()
    
    def evaluate_cost(pid_vector):
        pid = {'Kp': pid_vector[0], 'Ki': pid_vector[1], 'Kd': pid_vector[2]}
        sim_time = max(5, 4 * target_settling_time)
        try:
            settling_time, t, y, closed_loop, rise_time, overshoot = ph_agent.simulate_response(plant, pid, simulation_time=sim_time)
        except Exception as ex:
            return 1e6
        cost_ts = abs(settling_time - target_settling_time)
        cost_os = max(0, overshoot - target_overshoot)
        cost_rt = max(0, rise_time - target_rise_time) if rise_time is not None else 1e3
        reg_term = 0.001 * (pid['Kp']**2 + pid['Ki']**2 + pid['Kd']**2)
        return 1.0 * cost_ts + 0.1 * cost_os + 0.1 * cost_rt + reg_term

    for iter in range(max_iter):
        for i in range(swarm_size):
            cost = evaluate_cost(swarm_positions[i])
            if cost < pbest_costs[i]:
                pbest_costs[i] = cost
                pbest_positions[i] = swarm_positions[i].copy()
            if cost < gbest_cost:
                gbest_cost = cost
                gbest_position = swarm_positions[i].copy()
        for i in range(swarm_size):
            r1 = np.random.random(dim)
            r2 = np.random.random(dim)
            swarm_velocities[i] = (w * swarm_velocities[i] +
                                   c1 * r1 * (pbest_positions[i] - swarm_positions[i]) +
                                   c2 * r2 * (gbest_position - swarm_positions[i]))
            swarm_positions[i] = swarm_positions[i] + swarm_velocities[i]
            swarm_positions[i] = np.clip(swarm_positions[i], lb, ub)
        print(f"PSO Iteration {iter+1}/{max_iter}, Best Cost: {gbest_cost:.4f}")
    optimized_pid = {'Kp': gbest_position[0], 'Ki': gbest_position[1], 'Kd': gbest_position[2]}
    return optimized_pid

# ==============================
# Differential Evolution (DE) Optimization
# ==============================
def de_optimize_pid(plant, target_settling_time, target_overshoot, target_rise_time, initial_pid):
    lb = [0.1 * initial_pid['Kp'], 0.1 * initial_pid['Ki'], 0.1 * initial_pid['Kd']]
    ub = [100 * initial_pid['Kp'], 100 * initial_pid['Ki'], 100 * initial_pid['Kd']]
    def cost_func(x):
        pid = {'Kp': x[0], 'Ki': x[1], 'Kd': x[2]}
        sim_time = max(5, 4 * target_settling_time)
        try:
            settling_time, t, y, closed_loop, rise_time, overshoot = PyHesapAgent().simulate_response(plant, pid, simulation_time=sim_time)
        except Exception as ex:
            return 1e6
        cost_ts = abs(settling_time - target_settling_time)
        cost_os = max(0, overshoot - target_overshoot)
        cost_rt = max(0, rise_time - target_rise_time) if rise_time is not None else 1e3
        reg_term = 0.001 * (pid['Kp']**2 + pid['Ki']**2 + pid['Kd']**2)
        return 1.0 * cost_ts + 0.1 * cost_os + 0.1 * cost_rt + reg_term
    result = differential_evolution(cost_func, bounds=list(zip(lb, ub)), strategy='best1bin', maxiter=30, popsize=15)
    optimized_pid = {'Kp': result.x[0], 'Ki': result.x[1], 'Kd': result.x[2]}
    return optimized_pid

# ==============================
# PyHesap Agent: Calculation, Simulation, and Metric Computation
# ==============================
class PyHesapAgent:
    def simulate_response(self, plant, pid_params, simulation_time=5, dt=0.01):
        Kp = pid_params['Kp']
        Ki = pid_params['Ki']
        Kd = pid_params['Kd']
        pid_controller = tf([Kd, Kp, Ki], [1, 0])
        open_loop = series(pid_controller, plant)
        closed_loop = feedback(open_loop, 1)
        t = np.arange(0, simulation_time, dt)
        t, y = step_response(closed_loop, t)
        settling_time = self.compute_settling_time(t, y)
        rise_time = self.compute_rise_time(t, y)
        overshoot = self.compute_overshoot(y)
        return settling_time, t, y, closed_loop, rise_time, overshoot

    def compute_settling_time(self, t, y, tolerance=0.02):
        final_value = y[-1]
        lower_bound = final_value * (1 - tolerance)
        upper_bound = final_value * (1 + tolerance)
        settling_time = t[-1]
        for i in range(len(y)):
            if all(lower_bound <= y[j] <= upper_bound for j in range(i, len(y))):
                settling_time = t[i]
                break
        return settling_time

    def compute_rise_time(self, t, y):
        final_value = y[-1]
        start_time = None
        end_time = None
        for i, val in enumerate(y):
            if start_time is None and val >= 0.1 * final_value:
                start_time = t[i]
            if start_time is not None and val >= 0.9 * final_value:
                end_time = t[i]
                break
        if start_time is not None and end_time is not None:
            return end_time - start_time
        else:
            return None

    def compute_overshoot(self, y):
        final_value = y[-1]
        max_val = np.max(y)
        if final_value == 0:
            return 0
        return max(0, (max_val - final_value) / abs(final_value) * 100)

# ==============================
# Performance Comparison Frame
# (Orijinal haliyle – hiçbir değişiklik yapılmadı)
# ==============================
class PerformanceComparisonFrame(ttk.Frame):
    def __init__(self, master, performance_results):
        super().__init__(master)
        self.performance_results = performance_results
        self.create_widgets()
    def create_widgets(self):
        label = ttk.Label(self, text="Performance Comparison", font=("Helvetica", 14, "bold"))
        label.pack(pady=10)
        self.listbox = tk.Listbox(self, selectmode="multiple", width=80, height=10)
        self.listbox.pack(padx=10, pady=10, fill="both", expand=True)
        self.update_listbox()
        self.compare_button = ttk.Button(self, text="Compare", command=self.compare_runs)
        self.compare_button.pack(pady=5)
    def update_listbox(self):
        self.listbox.delete(0, tk.END)
        for i, res in enumerate(self.performance_results):
            summary = (f"Run {i+1}: PID={res['pid_params']}, "
                       f"Target={res['target_settling_time']}s, Final={res['final_settling_time']}s, "
                       f"Overshoot={res['overshoot']:.2f}%")
            self.listbox.insert(tk.END, summary)
    def compare_runs(self):
        selected_indices = list(self.listbox.curselection())
        if not selected_indices:
            messagebox.showwarning("Warning", "Please select at least one result for comparison.")
            return
        fig = plt.figure(figsize=(10,6))
        ax = fig.add_subplot(111)
        for idx in selected_indices:
            res = self.performance_results[idx]
            t = np.array(res['t'])
            y = np.array(res['y'])
            ax.plot(t, y, label=f"Run {idx+1}: PID={res['pid_params']}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Output Y(t)")
        ax.set_title("Performance Comparison")
        ax.legend()
        comp_window = tk.Toplevel(self)
        comp_window.title("Performance Comparison Graph")
        canvas = FigureCanvasTkAgg(fig, master=comp_window)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        toolbar = NavigationToolbar2Tk(canvas, comp_window)
        toolbar.update()
        canvas.draw()
        comp_window.protocol("WM_DELETE_WINDOW", lambda: comp_window.destroy())

# ==============================
# Interactive Design Window: Iterative PID Design + Real-Time Simulation
# ==============================
class InteractiveDesignWindow(tk.Toplevel):
    def __init__(self, master, plant, target_settling_time, on_design_complete,
                 target_overshoot, target_rise_time):
        super().__init__(master)
        self.title("PID Design Process")
        self.geometry("500x800")
        self.plant = plant
        self.target_overshoot = target_overshoot
        self.target_rise_time = target_rise_time
        # Save the initially provided target settling time both as initial_target_settling_time and target_settling_time.
        self.initial_target_settling_time = target_settling_time
        self.target_settling_time = target_settling_time
        self.on_design_complete = on_design_complete
        self.pid_params = {'Kp': 1.0, 'Ki': 0.1, 'Kd': 0.01}
        self.optimization_method = tk.StringVar(value="PSO")
        self.optimized_metrics = {}
        self.create_widgets()
    
    def create_widgets(self):
        # Optimization section
        opt_frame = ttk.Frame(self)
        opt_frame.pack(pady=5, fill="x")
        ttk.Label(opt_frame, text="Optimization Method:", font=("Helvetica", 10, "bold")).pack(side="left", padx=5)
        ttk.Radiobutton(opt_frame, text="PSO", variable=self.optimization_method, value="PSO", style="TRadiobutton").pack(side="left", padx=5)
        ttk.Radiobutton(opt_frame, text="Differential Evolution", variable=self.optimization_method, value="DE", style="TRadiobutton").pack(side="left", padx=5)
        self.btn_opt = ttk.Button(self, text="Run Optimization", command=self.run_optimization)
        self.btn_opt.pack(pady=5)
        ToolTip(self.btn_opt, "Optimize PID parameters using the selected optimization algorithm.")
        self.result_label = ttk.Label(self, text="Optimization result not yet calculated.", font=("Helvetica", 12))
        self.result_label.pack(pady=5)
        self.btn_show_graph = ttk.Button(self, text="Show Graph", command=self.show_results, state="disabled")
        self.btn_show_graph.pack(pady=5)
        ToolTip(self.btn_show_graph, "Display the optimization results on a graph.")
        self.btn_accept = ttk.Button(self, text="Accept Design", command=self.accept_design, state="disabled")
        self.btn_accept.pack(pady=5)
        ToolTip(self.btn_accept, "Accept the optimization results and complete the PID design process.")
        
        # Real-Time Simulation with manual entry fields added next to sliders.
        rt_frame = ttk.LabelFrame(self, text="Real-Time Simulation", style="TLabelframe")
        rt_frame.pack(pady=5, fill="both", expand=True)
        
        # Kp slider and entry
        kp_frame = ttk.Frame(rt_frame)
        kp_frame.pack(fill="x", padx=5, pady=2)
        self.kp_slider = tk.Scale(kp_frame, from_=0.0, to=1000.0, resolution=0.01, orient="horizontal", label="Kp", command=self.update_realtime_simulation)
        self.kp_slider.set(self.pid_params['Kp'])
        self.kp_slider.pack(side="left", fill="x", expand=True)
        self.kp_entry = ttk.Entry(kp_frame, width=10)
        self.kp_entry.insert(0, str(self.pid_params['Kp']))
        self.kp_entry.pack(side="left", padx=5)
        self.kp_entry.bind("<Return>", self.manual_kp_update)
        self.kp_entry.bind("<FocusOut>", self.manual_kp_update)
        
        # Ki slider and entry
        ki_frame = ttk.Frame(rt_frame)
        ki_frame.pack(fill="x", padx=5, pady=2)
        self.ki_slider = tk.Scale(ki_frame, from_=0.0, to=1000.0, resolution=0.01, orient="horizontal", label="Ki", command=self.update_realtime_simulation)
        self.ki_slider.set(self.pid_params['Ki'])
        self.ki_slider.pack(side="left", fill="x", expand=True)
        self.ki_entry = ttk.Entry(ki_frame, width=10)
        self.ki_entry.insert(0, str(self.pid_params['Ki']))
        self.ki_entry.pack(side="left", padx=5)
        self.ki_entry.bind("<Return>", self.manual_ki_update)
        self.ki_entry.bind("<FocusOut>", self.manual_ki_update)
        
        # Kd slider and entry
        kd_frame = ttk.Frame(rt_frame)
        kd_frame.pack(fill="x", padx=5, pady=2)
        self.kd_slider = tk.Scale(kd_frame, from_=0.0, to=1000.0, resolution=0.01, orient="horizontal", label="Kd", command=self.update_realtime_simulation)
        self.kd_slider.set(self.pid_params['Kd'])
        self.kd_slider.pack(side="left", fill="x", expand=True)
        self.kd_entry = ttk.Entry(kd_frame, width=10)
        self.kd_entry.insert(0, str(self.pid_params['Kd']))
        self.kd_entry.pack(side="left", padx=5)
        self.kd_entry.bind("<Return>", self.manual_kd_update)
        self.kd_entry.bind("<FocusOut>", self.manual_kd_update)
        
        self.rt_confirm_button = ttk.Button(rt_frame, text="Confirm New PID", command=self.confirm_rt_parameters)
        self.rt_confirm_button.pack(pady=5)
        self.rt_fig = mplfig.Figure(figsize=(5,3))
        self.rt_ax = self.rt_fig.add_subplot(111)
        self.rt_canvas = FigureCanvasTkAgg(self.rt_fig, master=rt_frame)
        self.rt_canvas.get_tk_widget().pack(fill="both", expand=True, padx=5, pady=5)
        self.update_realtime_simulation(None)
    
    def update_realtime_simulation(self, event):
        kp = self.kp_slider.get()
        ki = self.ki_slider.get()
        kd = self.kd_slider.get()
        self.kp_entry.delete(0, tk.END)
        self.kp_entry.insert(0, str(kp))
        self.ki_entry.delete(0, tk.END)
        self.ki_entry.insert(0, str(ki))
        self.kd_entry.delete(0, tk.END)
        self.kd_entry.insert(0, str(kd))
        pid = {"Kp": kp, "Ki": ki, "Kd": kd}
        ph_agent = PyHesapAgent()
        # İlk simülasyon süresi hedef settling time’ın 4 katı, en az 5 s olarak belirleniyor.
        sim_time = max(5, 4 * self.initial_target_settling_time)
        settling_time, t, y, closed_loop, rise_time, overshoot = ph_agent.simulate_response(self.plant, pid, simulation_time=sim_time)
        self.rt_ax.clear()
        self.rt_ax.plot(t, y, 'b-', label="System Response")
        self.rt_ax.axhline(y=1, color='k', linestyle=':', label="Target")
        self.rt_ax.set_xlabel("Time (s)")
        self.rt_ax.set_ylabel("Output Y(t)")
        self.rt_ax.legend()
        # x eksenini, en son elde edilen settling time'a göre güncelle: max(5, 4 * computed_settling_time)
        new_sim_time = max(5, 4 * settling_time)
        self.rt_ax.set_xlim(0, new_sim_time)
        self.rt_fig.tight_layout()
        self.rt_canvas.draw()
    
    def manual_kp_update(self, event):
        try:
            value = float(self.kp_entry.get())
        except ValueError:
            value = self.pid_params['Kp']
        self.kp_slider.set(value)
        self.update_realtime_simulation(None)
    
    def manual_ki_update(self, event):
        try:
            value = float(self.ki_entry.get())
        except ValueError:
            value = self.pid_params['Ki']
        self.ki_slider.set(value)
        self.update_realtime_simulation(None)
    
    def manual_kd_update(self, event):
        try:
            value = float(self.kd_entry.get())
        except ValueError:
            value = self.pid_params['Kd']
        self.kd_slider.set(value)
        self.update_realtime_simulation(None)
    
    def confirm_rt_parameters(self):
        kp = self.kp_slider.get()
        ki = self.ki_slider.get()
        kd = self.kd_slider.get()
        self.pid_params = {"Kp": kp, "Ki": ki, "Kd": kd}
        ph_agent = PyHesapAgent()
        sim_time = max(5, 4 * self.initial_target_settling_time)
        settling_time, t, y, closed_loop, rise_time, overshoot = ph_agent.simulate_response(self.plant, self.pid_params, simulation_time=sim_time)
        self.optimized_metrics = {
            "settling_time": settling_time,
            "rise_time": rise_time,
            "overshoot": overshoot
        }
        result_text = (f"Manual PID Result:\n"
                       f"Kp = {self.pid_params['Kp']:.4f}\n"
                       f"Ki = {self.pid_params['Ki']:.4f}\n"
                       f"Kd = {self.pid_params['Kd']:.4f}\n"
                       f"Settling Time = {settling_time:.3f} s\n"
                       f"Rise Time = {rise_time:.3f} s\n"
                       f"Max Overshoot = {overshoot:.2f}%")
        self.result_label.config(text=result_text)
        self.btn_show_graph.config(state="normal")
        self.btn_accept.config(state="normal")
        self.t = t
        self.y = y
        self.closed_loop = closed_loop
    
    def run_optimization(self):
        initial_pid = self.pid_params
        if self.optimization_method.get() == "PSO":
            optimized_pid = pso_optimize_pid(self.plant, self.initial_target_settling_time,
                                             self.target_overshoot, self.target_rise_time,
                                             initial_pid)
        else:
            optimized_pid = de_optimize_pid(self.plant, self.initial_target_settling_time,
                                            self.target_overshoot, self.target_rise_time,
                                            initial_pid)
        self.pid_params = optimized_pid
        ph_agent = PyHesapAgent()
        sim_time = max(5, 4 * self.initial_target_settling_time)
        settling_time, t, y, closed_loop, rise_time, overshoot = ph_agent.simulate_response(self.plant, self.pid_params, simulation_time=sim_time)
        self.optimized_metrics = {
            "settling_time": settling_time,
            "rise_time": rise_time,
            "overshoot": overshoot
        }
        result_text = (f"Optimization Result:\n"
                       f"Kp = {self.pid_params['Kp']:.4f}\n"
                       f"Ki = {self.pid_params['Ki']:.4f}\n"
                       f"Kd = {self.pid_params['Kd']:.4f}\n"
                       f"Settling Time = {settling_time:.3f} s\n"
                       f"Rise Time = {rise_time:.3f} s\n"
                       f"Max Overshoot = {overshoot:.2f}%")
        self.result_label.config(text=result_text)
        self.btn_show_graph.config(state="normal")
        self.btn_accept.config(state="normal")
        self.kp_slider.set(self.pid_params['Kp'])
        self.ki_slider.set(self.pid_params['Ki'])
        self.kd_slider.set(self.pid_params['Kd'])
        self.update_realtime_simulation(None)
        self.t = t
        self.y = y
        self.closed_loop = closed_loop

    def generate_result_figure(self):
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        ax.plot(self.t, self.y, 'b-', linewidth=2, label="System Response (Y(t))")
        ax.set_xlabel("Time (s)", fontsize=12)
        ax.set_ylabel("Output Y(t)", fontsize=12)
        ax.set_title("Final Closed-Loop Step Response", fontsize=14)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.minorticks_on()
        ax.axhline(y=1, color='k', linestyle=':', linewidth=1.5, label="Target Output (1)")
        settle_idx = None
        for i in range(len(self.y)):
            if all(0.98 <= self.y[j] <= 1.02 for j in range(i, len(self.y))):
                settle_idx = i
                break
        if settle_idx is not None:
            ax.plot(self.t[settle_idx], self.y[settle_idx], 'ro', markersize=8, label="Settling Point")
            ax.axvline(x=self.t[settle_idx], color='r', linestyle='--', linewidth=1)
        start_idx = next((i for i, val in enumerate(self.y) if val >= 0.1), None)
        end_idx = next((i for i, val in enumerate(self.y) if val >= 0.9), None)
        if start_idx is not None and end_idx is not None:
            ax.plot(self.t[start_idx], self.y[start_idx], 'go', markersize=8, label="Rise Start")
            ax.plot(self.t[end_idx], self.y[end_idx], 'mo', markersize=8, label="Rise End")
            ax.axvline(x=self.t[start_idx], color='g', linestyle='--', linewidth=1)
            ax.axvline(x=self.t[end_idx], color='m', linestyle='--', linewidth=1)
        max_idx = np.argmax(self.y)
        if self.y[max_idx] > 1:
            ax.plot(self.t[max_idx], self.y[max_idx], 'co', markersize=8, label="Maximum Overshoot Point")
            ax.axhline(y=self.y[max_idx], color='c', linestyle='--', linewidth=1)
        ax.axvline(x=self.master.target_settling_time, color='b', linestyle='--', linewidth=1.5, label="Target Settling Time")
        pid_controller = tf([self.pid_params['Kd'], self.pid_params['Kp'], self.pid_params['Ki']], [1, 0])
        L = series(pid_controller, tf([1], [1,2,1]))
        gm, pm, wg, wp = margin(L)
        rise_time_str = f"{self.optimized_metrics['rise_time']:.3f}" if self.optimized_metrics['rise_time'] is not None else "Not determined"
        annot_text = (
            f"Settling Time: {self.optimized_metrics['settling_time']:.3f} s\n"
            f"Rise Time: {rise_time_str} s\n"
            f"Max Overshoot: {self.optimized_metrics['overshoot']:.2f}%\n"
            f"Phase Margin: {pm:.2f}°\n"
            f"Gain Margin: {'Infinite' if gm==np.inf else f'{gm:.2f} dB'}"
        )
        ax.text(0.65, 0.80, annot_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7))
        ax.legend(loc="best")
        fig.tight_layout()
        return fig

    def show_results(self):
        fig = self.generate_result_figure()
        def on_click(event):
            if event.inaxes is not None:
                annotation_text = simpledialog.askstring("Add Label", "Enter label text:")
                if annotation_text:
                    ax = fig.get_axes()[0]
                    ax.annotate(annotation_text, xy=(event.xdata, event.ydata),
                                xytext=(event.xdata + 0.1, event.ydata + 0.1),
                                arrowprops=dict(facecolor='black', arrowstyle='->'))
                    canvas.draw()
        canvas = FigureCanvasTkAgg(fig, master=self.master)
        fig.canvas.mpl_connect("button_press_event", on_click)
        
        graph_window = tk.Toplevel(self)
        graph_window.title("Graph Preview")
        canvas = FigureCanvasTkAgg(fig, master=graph_window)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        toolbar = NavigationToolbar2Tk(canvas, graph_window)
        toolbar.update()
        canvas.draw()
        graph_window.protocol("WM_DELETE_WINDOW", lambda: self.on_graph_window_closed(fig, graph_window))
    
    def on_graph_window_closed(self, fig, graph_window):
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        self.master.saved_graph_bytes = buf.getvalue()
        photo = ImageTk.PhotoImage(Image.open(buf))
        self.master.saved_graph_photo = photo
        self.master.update_graph_preview(photo)
        graph_window.destroy()
    
    def accept_design(self):
        rise_time_str = f"{self.optimized_metrics['rise_time']:.3f}" if self.optimized_metrics['rise_time'] is not None else "N/A"
        msg = (
            f"Design accepted!\nFinal PID: {self.pid_params}\n"
            f"Final Settling Time: {self.optimized_metrics['settling_time']:.3f} s (Measured)\n"
            f"Target Settling Time: {self.master.target_settling_time:.3f} s (Initial)\n"
            f"Final Rise Time: {rise_time_str} s (Target: {self.target_rise_time} s)\n"
            f"Final Max Overshoot: {self.optimized_metrics['overshoot']:.2f}% (Target: {self.target_overshoot}%)"
        )
        messagebox.showinfo("Design Accepted", msg)
        fig = self.generate_result_figure()
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        photo = ImageTk.PhotoImage(Image.open(buf))
        self.master.saved_graph_bytes = buf.getvalue()
        self.master.saved_graph_photo = photo
        self.master.update_graph_preview(photo)
        self.show_results()
        self.on_design_complete(self.plant, self.pid_params, self.master.target_settling_time)
        self.destroy()
    
    def on_design_complete(self, plant, pid_params, target_settling_time):
        ph_agent = PyHesapAgent()
        sim_time = max(5, 4 * target_settling_time)
        settling_time, t, y, closed_loop, rise_time, overshoot = ph_agent.simulate_response(plant, pid_params, simulation_time=sim_time)
        self.final_pid_params = pid_params
        self.final_settling_time = settling_time
        self.final_plant = plant
        result_dict = {
            "pid_params": pid_params,
            "target_settling_time": target_settling_time,
            "final_settling_time": settling_time,
            "rise_time": rise_time,
            "overshoot": overshoot,
            "t": t.tolist(),
            "y": y.tolist()
        }
        self.performance_results.append(result_dict)
        self.performance_comparison_frame.update_listbox()
        for widget in self.result_eval_frame.winfo_children():
            widget.destroy()
        result_frame = ResultEvaluationFrame(self.result_eval_frame, self.final_pid_params, settling_time, target_settling_time, self.final_plant)
        result_frame.pack(fill="both", expand=True)
        self.notebook.select(self.result_eval_frame)
        self.graph_preview_locked = True

# ==============================
# Project Save and Load Panel
# ==============================
class ProjectManagerFrame(ttk.Frame):
    def __init__(self, master, app):
        super().__init__(master)
        self.app = app  # Reference to MainApplication
        self.create_widgets()
    def create_widgets(self):
        label = ttk.Label(self, text="Projects", font=("Helvetica", 14, "bold"))
        label.pack(pady=10)
        self.save_button = ttk.Button(self, text="Save Project", command=self.save_project)
        self.save_button.pack(pady=5)
        self.load_button = ttk.Button(self, text="Load Project", command=self.load_project)
        self.load_button.pack(pady=5)
        self.project_text = scrolledtext.ScrolledText(self, width=80, height=15, font=("Helvetica", 10))
        self.project_text.pack(padx=10, pady=10, fill="both", expand=True)
        self.graph_label = ttk.Label(self)
        self.graph_label.pack(pady=5)
    def save_project(self):
        if self.app.final_pid_params is None:
            messagebox.showwarning("Warning", "No project design has been accepted yet.")
            return
        project_data = {
            "final_pid_params": self.app.final_pid_params,
            "target_settling_time": self.app.target_settling_time,
            "target_overshoot": self.app.target_overshoot,
            "target_rise_time": self.app.target_rise_time
        }
        ph_agent = PyHesapAgent()
        settling_time, t, y, closed_loop, rise_time, overshoot = ph_agent.simulate_response(self.app.final_plant, self.app.final_pid_params)
        fig = plt.figure(figsize=(6,4))
        plt.plot(t, y, 'b-', linewidth=2, label="System Response (Y(t))")
        plt.xlabel("Time (s)")
        plt.ylabel("Output Y(t)")
        plt.title("Final Closed-Loop Step Response")
        plt.axhline(y=1, color='k', linestyle=':', linewidth=1.5, label="Target")
        plt.legend()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
        image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        project_data["simulation_graph"] = image_base64
        file_path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON files","*.json")], title="Save Project")
        if file_path:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(project_data, f, indent=4)
            messagebox.showinfo("Info", "Project saved successfully.")
    def load_project(self):
        file_path = filedialog.askopenfilename(filetypes=[("JSON files","*.json")], title="Load Project")
        if file_path:
            with open(file_path, "r", encoding="utf-8") as f:
                project_data = json.load(f)
            self.app.final_pid_params = project_data.get("final_pid_params")
            self.app.target_settling_time = project_data.get("target_settling_time")
            self.app.target_overshoot = project_data.get("target_overshoot")
            self.app.target_rise_time = project_data.get("target_rise_time")
            messagebox.showinfo("Info", "Project loaded and design accepted.")
            self.project_text.delete("1.0", tk.END)
            summary = f"PID: {self.app.final_pid_params}\n" \
                      f"Target Settling Time: {self.app.target_settling_time}\n" \
                      f"Target Overshoot: {self.app.target_overshoot}\n" \
                      f"Target Rise Time: {self.app.target_rise_time}"
            self.project_text.insert(tk.END, summary)
            image_base64 = project_data.get("simulation_graph")
            if image_base64:
                image_data = base64.b64decode(image_base64)
                image = Image.open(io.BytesIO(image_data))
                image.thumbnail((600, 400), Image.ANTIALIAS)
                photo = ImageTk.PhotoImage(image)
                self.graph_label.config(image=photo)
                self.graph_label.image = photo

# ==============================
# General NLP Chat Frame
# ==============================
class GeneralChatFrame(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.create_widgets()
    def create_widgets(self):
        header = ttk.Label(self, text="General Chat (This section is for general discussion)", font=("Helvetica", 12, "bold"))
        header.pack(pady=5)
        self.chat_history = scrolledtext.ScrolledText(self, state='disabled', width=80, height=20, font=("Helvetica", 10))
        self.chat_history.pack(padx=10, pady=10, fill="both", expand=True)
        input_frame = ttk.Frame(self)
        input_frame.pack(padx=10, pady=5, fill="x")
        self.user_input = ttk.Entry(input_frame, width=70, font=("Helvetica", 10))
        self.user_input.pack(side="left", padx=(0,5), fill="x", expand=True)
        self.send_button = ttk.Button(input_frame, text="Send", command=self.send_message)
        self.send_button.pack(side="left")
        ToolTip(self.send_button, "Send your message. (Replies will be in English.)")
        self.user_input.bind("<Return>", self.send_message)
    def send_message(self, event=None):
        message = self.user_input.get().strip()
        if message == "": return
        self.append_message("User", message)
        self.user_input.delete(0, tk.END)
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "system", "content": "You are a helpful assistant that answers in English."},
                          {"role": "user", "content": message}]
            )
            reply = response.choices[0].message.content.strip()
        except Exception as e:
            reply = "Error: " + str(e)
        self.append_message("Assistant", reply)
    def append_message(self, sender, message):
        self.chat_history.configure(state='normal')
        self.chat_history.insert(tk.END, f"{sender}: {message}\n")
        self.chat_history.configure(state='disabled')
        self.chat_history.see(tk.END)

# ==============================
# Result Evaluation Assistant Frame
# ==============================
class ResultEvaluationFrame(ttk.Frame):
    def __init__(self, master, final_pid_params, final_settling_time, target_settling_time, final_plant):
        super().__init__(master)
        self.final_pid_params = final_pid_params
        self.final_settling_time = final_settling_time
        self.target_settling_time = target_settling_time
        self.final_plant = final_plant
        self.create_widgets()
    def create_widgets(self):
        header = ttk.Label(self, text="Result Evaluation (Optimization Results)", font=("Helvetica", 12, "bold"))
        header.pack(pady=5)
        summary = (
            f"PID: {self.final_pid_params}\n"
            f"Target Settling Time: {self.target_settling_time:.3f} s\n"
            f"Final Settling Time: {self.final_settling_time:.3f} s\n"
            f"Error: {abs(self.final_settling_time - self.target_settling_time):.3f} s\n"
        )
        summary_label = ttk.Label(self, text=summary, font=("Courier", 11), background="#e0f7fa")
        summary_label.pack(padx=10, pady=5, fill="x")
        
        self.btn_generate_matlab = ttk.Button(self, text="Generate MATLAB Code", command=self.generate_matlab_code)
        self.btn_generate_matlab.pack(pady=5)
        ToolTip(self.btn_generate_matlab, "Generate MATLAB code using the final design data.")
        
        self.matlab_code_box = scrolledtext.ScrolledText(self, width=80, height=10, font=("Helvetica", 10))
        self.matlab_code_box.pack(padx=10, pady=5, fill="both", expand=True)
        
        self.btn_generate_report = ttk.Button(self, text="Generate Report (PDF)", command=self.generate_report)
        self.btn_generate_report.pack(pady=5)
        ToolTip(self.btn_generate_report, "Generate a PDF report containing final design data and graph preview.")
        
        self.chat_history = scrolledtext.ScrolledText(self, state='disabled', width=80, height=10, font=("Helvetica", 10))
        self.chat_history.pack(padx=10, pady=10, fill="both", expand=True)
        info = (
            "Project: SmartControl\n"
            "Your results are listed above. Please enter your questions for evaluation."
        )
        self.append_message("System", info)
        input_frame = ttk.Frame(self)
        input_frame.pack(padx=10, pady=5, fill="x")
        self.user_input = ttk.Entry(input_frame, width=70, font=("Helvetica", 10))
        self.user_input.pack(side="left", padx=(0,5), fill="x", expand=True)
        self.send_button = ttk.Button(input_frame, text="Send", command=self.send_message)
        self.send_button.pack(side="left")
        ToolTip(self.send_button, "Send your questions for evaluation.")
        self.user_input.bind("<Return>", self.send_message)
    
    def generate_matlab_code(self):
        pid = self.final_pid_params
        try:
            num_str = ','.join(map(str, self.final_plant.num[0]))
            den_str = ','.join(map(str, self.final_plant.den[0]))
        except Exception as e:
            num_str = "1"
            den_str = "1,2,1"
        plant_str = f"tf([{num_str}], [{den_str}])"
        prompt = (
            "Please generate MATLAB code that performs the following tasks, ensuring the output is visually appealing and easy for a control engineer to analyze:\n"
            "1. Create a closed-loop control system using the given PID parameters and transfer function.\n"
            "2. Simulate the system's step response and produce a well-formatted plot.\n"
            "3. The plot should include:\n"
            "   - A clear title that indicates the system performance and target settling time.\n"
            "   - Properly labeled axes (e.g., Time in seconds, System Output).\n"
            "   - Grid lines for enhanced readability.\n"
            "   - A legend differentiating the system response and the reference signal.\n"
            "   - Annotations or markers showing key performance metrics such as the target settling time and overshoot if applicable.\n"
            "4. Include detailed comments throughout the code to explain each section for clarity and easy interpretation by a control engineer.\n"
            "5. Ensure the script is fully executable in MATLAB as a standalone file.\n\n"
            f"PID: {pid}\n"
            f"Transfer Function: {plant_str}\n"
            f"Target Settling Time: {self.target_settling_time} s\n"
        )
        def get_matlab_code():
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[{"role": "system", "content": "Generate clear, well-commented, and visually appealing MATLAB code as described."},
                            {"role": "user", "content": prompt}]
                )
                matlab_code = response.choices[0].message.content.strip()
            except Exception as e:
                matlab_code = f"% Error generating MATLAB code: {e}"
            self.after(0, lambda: self.set_matlab_code(matlab_code))
        threading.Thread(target=get_matlab_code).start()
        
    def set_matlab_code(self, code):
        self.matlab_code_box.delete("1.0", tk.END)
        self.matlab_code_box.insert(tk.END, code)
    def generate_report(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF files", "*.pdf")], title="Save Report")
        if not file_path:
            return
        c = canvas.Canvas(file_path, pagesize=letter)
        width, height = letter
        textobject = c.beginText(40, height - 40)
        textobject.setFont("Helvetica", 12)
        textobject.textLine("SmartControl PID Design Report")
        textobject.textLine("")
        textobject.textLine(f"PID Parameters: {self.final_pid_params}")
        textobject.textLine(f"Target Settling Time: {self.target_settling_time} s")
        textobject.textLine(f"Final Settling Time: {self.final_settling_time} s")
        error = abs(self.final_settling_time - self.target_settling_time)
        textobject.textLine(f"Error: {error} s")
        c.drawText(textobject)
        main_app = self.winfo_toplevel()
        if hasattr(main_app, "saved_graph_bytes"):
            temp_file = "temp_graph.png"
            with open(temp_file, "wb") as f:
                f.write(main_app.saved_graph_bytes)
            c.drawImage(temp_file, 40, height/2 - 100, width=width-80, preserveAspectRatio=True, mask='auto')
            os.remove(temp_file)
        c.save()
        messagebox.showinfo("Report", "Report generated successfully.")
    def send_message(self, event=None):
        message = self.user_input.get().strip()
        if message == "": return
        self.append_message("User", message)
        self.user_input.delete(0, tk.END)
        system_message = (
            "The following PID results are provided:\n"
            f"PID: {self.final_pid_params}\n"
            f"Final Settling Time: {self.final_settling_time:.3f} s\n"
            f"Target Settling Time: {self.target_settling_time:.3f} s\n"
            "Please evaluate these results."
        )
        def get_reply():
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[{"role": "system", "content": system_message},
                              {"role": "user", "content": message}]
                )
                reply = response.choices[0].message.content.strip()
            except Exception as e:
                reply = "Error: " + str(e)
            self.after(0, lambda: self.append_message("Assistant", reply))
        threading.Thread(target=get_reply).start()
    def append_message(self, sender, message):
        self.chat_history.configure(state='normal')
        self.chat_history.insert(tk.END, f"{sender}: {message}\n")
        self.chat_history.configure(state='disabled')
        self.chat_history.see(tk.END)

# ==============================
# Feedback Frame
# ==============================
class FeedbackFrame(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.create_widgets()
    def create_widgets(self):
        lbl_title = ttk.Label(self, text="Feedback", font=("Helvetica", 16, "bold"))
        lbl_title.pack(pady=10)
        self.feedback_text = scrolledtext.ScrolledText(self, width=80, height=10, font=("Helvetica", 12))
        self.feedback_text.pack(padx=10, pady=10, fill="both", expand=True)
        input_frame = ttk.Frame(self)
        input_frame.pack(padx=10, pady=5, fill="x")
        self.user_feedback = ttk.Entry(input_frame, width=70, font=("Helvetica", 12))
        self.user_feedback.pack(side="left", padx=(0,5), fill="x", expand=True)
        self.send_button = ttk.Button(input_frame, text="Send", command=self.send_feedback)
        self.send_button.pack(side="left")
        ToolTip(self.send_button, "Submit your feedback.")
        self.user_feedback.bind("<Return>", self.send_feedback)
    def send_feedback(self, event=None):
        feedback = self.user_feedback.get().strip()
        if feedback == "": return
        try:
            with open("feedback.txt", "a", encoding="utf-8") as f:
                f.write(feedback + "\n")
        except Exception as e:
            messagebox.showerror("Error", f"Could not save feedback: {e}")
            return
        print("Feedback:", feedback)
        messagebox.showinfo("Thank You", "Your feedback has been recorded.")
        self.user_feedback.delete(0, tk.END)

# ==============================
# Code Backup Function
# ==============================
def backup_code():
    try:
        try:
            current_file = __file__
        except NameError:
            print("Backup: __file__ is not defined. Code backup cannot be taken.")
            return
        with open(current_file, "r", encoding="utf-8") as f:
            code_contents = f.read()
        now = datetime.datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        backup_filename = f"smartcontrol_backup_{timestamp}.py"
        with open(backup_filename, "w", encoding="utf-8") as f:
            f.write(code_contents)
        print(f"Code backup taken: {backup_filename}")
    except Exception as e:
        print(f"Error while taking code backup: {e}")

# ==============================
# Theme and Language Settings Window (English Themes)
# ==============================
def open_theme_settings(master, style):
    win = tk.Toplevel(master)
    win.title("Theme and Language Settings")
    win.geometry("400x300")
    tk.Label(win, text="Select Theme:", font=("Helvetica", 12, "bold")).pack(pady=10)
    themes = {"Modern": "clam", "Old": "alt", "Default": "default", "Classic": "classic"}
    theme_var = tk.StringVar(value=[k for k, v in themes.items() if v == style.theme_use()][0])
    for theme_name in themes:
        tk.Radiobutton(win, text=theme_name, variable=theme_var, value=theme_name, font=("Helvetica", 10)).pack(anchor="w", padx=20)
    tk.Label(win, text="(Language options can be added)", font=("Helvetica", 10, "italic")).pack(pady=20)
    def apply_settings():
        try:
            style.theme_use(themes[theme_var.get()])
            messagebox.showinfo("Theme and Language", "Settings applied!")
        except Exception as e:
            messagebox.showerror("Error", f"Could not apply settings: {e}")
        win.destroy()
    tk.Button(win, text="Apply", command=apply_settings, font=("Helvetica", 10, "bold")).pack(pady=10)

# ==============================
# FAQ / Help Window
# ==============================
def open_faq(master):
    faq_win = tk.Toplevel(master)
    faq_win.title("Frequently Asked Questions (FAQ) / Help")
    faq_win.geometry("600x400")
    faq_text = (
        "Question 1: What is PID control?\n"
        "Answer: PID control is a strategy that uses proportional, integral, and derivative components to respond to the error signal.\n\n"
        "Question 2: What is a transfer function?\n"
        "Answer: A transfer function is a mathematical representation of a system's input-output relationship in terms of the Laplace transform.\n\n"
        "Question 3: How do optimization algorithms work?\n"
        "Answer: PSO and Differential Evolution optimize the PID parameters based on the system performance criteria.\n\n"
        "Question 4: How is MATLAB code generated?\n"
        "Answer: MATLAB code is automatically generated using the final PID parameters and transfer function information.\n\n"
        "Question 5: How do I save/load a project?\n"
        "Answer: The designed PID, target values, and simulation graph are saved as a JSON file and can be reloaded later."
    )
    st = scrolledtext.ScrolledText(faq_win, wrap=tk.WORD, font=("Helvetica", 12))
    st.insert(tk.END, faq_text)
    st.configure(state="disabled")
    st.pack(fill="both", expand=True, padx=10, pady=10)
    tk.Button(faq_win, text="Close", command=faq_win.destroy, font=("Helvetica", 10, "bold")).pack(pady=10)

# ==============================
# Main Application Window
# ==============================
class MainApplication(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("SmartControl PID Design")
        self.geometry("1100x800")
        self.configure(bg="#e6f2ff")
        self.resizable(True, True)
        self.style = ttk.Style(self)
        self.style.theme_use("clam")
        self.style.configure("TLabel", background="#e6f2ff", foreground="#003366", font=("Helvetica", 12))
        self.style.configure("TButton", background="#cce6ff", foreground="#003366", font=("Helvetica", 12, "bold"), borderwidth=2, relief="raised")
        self.style.map("TButton", background=[('active', '#99ccff')], relief=[('active', 'sunken')])
        self.style.configure("TFrame", background="#e6f2ff")
        self.style.configure("TEntry", fieldbackground="white", foreground="black")
        self.style.configure("TNotebook", background="#e6f2ff")
        self.style.configure("TNotebook.Tab", background="#cce6ff", foreground="#003366", font=("Helvetica", 12, "bold"), padding=[10, 5])
        self.style.configure("TRadiobutton", background="#e6f2ff", foreground="#003366", font=("Helvetica", 12, "bold"))
        self.style.configure("TLabelframe", background="#e6f2ff", relief="groove", borderwidth=2)
        
        self.load_logo()
        
        self.final_pid_params = None
        self.final_settling_time = None
        self.target_settling_time = None
        self.target_overshoot = None
        self.target_rise_time = None
        self.final_plant = None
        self.saved_graph_bytes = None
        self.saved_graph_photo = None
        self.performance_results = []  # List to store all run results
        self.graph_preview_locked = False
        
        self.create_menu()
        self.create_notebook()
        self.create_footer()
    
    def load_logo(self):
        try:
            logo_img = Image.open("CS-NLP_Logo.png")
            logo_img = logo_img.resize((150, 150), Image.ANTIALIAS)
            self.logo_image = ImageTk.PhotoImage(logo_img)
            logo_frame = ttk.Frame(self, style="TFrame")
            logo_frame.pack(side="top", fill="x", pady=10)
            self.logo_label = ttk.Label(logo_frame, image=self.logo_image, background="#e6f2ff")
            self.logo_label.pack(anchor="center")
        except Exception as e:
            print("Logo could not be loaded:", e)
    
    def create_menu(self):
        menubar = tk.Menu(self)
        self.config(menu=menubar)
        file_menu = tk.Menu(menubar, tearoff=False)
        file_menu.add_command(label="Select Control Function", 
            command=lambda: messagebox.showinfo("Info", "PID parameters will be automatically adjusted using optimization algorithms.\n\nBoth manual and sample options are available for entering the transfer function."))
        file_menu.add_command(label="Exit", command=self.quit)
        menubar.add_cascade(label="File", menu=file_menu)
        
        help_menu = tk.Menu(menubar, tearoff=False)
        help_menu.add_command(label="About", command=self.show_about)
        help_menu.add_command(label="Documentation", command=self.show_documentation)
        
        settings_menu = tk.Menu(menubar, tearoff=False)
        settings_menu.add_command(label="Theme and Language", command=lambda: open_theme_settings(self, self.style))
        settings_menu.add_command(label="FAQ / Help", command=lambda: open_faq(self))
        settings_menu.add_command(label="Support and Contact", 
            command=lambda: messagebox.showinfo("Support and Contact", "For support: kadir.tohma@iste.edu.tr\nCS NLP Group:\nProf. Dr. Celaleddin Yeroğlu\nDr. Lecturer Kadir Tohma\nDr. Lecturer Handan Gürsoy Demir\nDr. Lecturer Halil İbrahim Okur\nDr. Lecturer Merve Nilay Aydın"))
        menubar.add_cascade(label="Settings", menu=settings_menu)
        menubar.add_cascade(label="Help", menu=help_menu)
    
    def show_about(self):
        info_text = (
            "System: SmartControl PID Design\n\n"
            "This project automates control system design using LLMs and Python-based computational modules.\n\n"
            "Features:\n"
            " - PID control design\n"
            " - Optimization using PSO and Differential Evolution\n"
            " - MATLAB-style graph display\n"
            " - Project saving/loading\n"
            " - Real-time simulation: instant monitoring using PID parameter sliders and confirmation button\n\n"
            "Additionally, performance comparison and user feedback help improve the design."
        )
        messagebox.showinfo("About", info_text)
    
    def show_documentation(self):
        doc_text = (
            "SmartControl is an application designed to automate control system design.\n\n"
            "Main Features:\n"
            " - Extracting target performance criteria from natural language descriptions\n"
            " - Iterative PID control design and optimization (PSO, Differential Evolution)\n"
            " - MATLAB-style graph display\n"
            " - Project saving/loading\n"
            " - Real-time simulation: instant monitoring using PID sliders and confirmation button\n"
            " - Performance Comparison: Each run result is stored and can be compared via graphs.\n\n"
            "Additional documentation can be provided with further resources."
        )
        doc_window = tk.Toplevel(self)
        doc_window.title("Documentation")
        doc_window.geometry("600x400")
        txt = scrolledtext.ScrolledText(doc_window, wrap=tk.WORD, font=("Helvetica", 12))
        txt.insert(tk.END, doc_text)
        txt.configure(state="disabled")
        txt.pack(fill="both", expand=True, padx=10, pady=10)
    
    def create_notebook(self):
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=10)
        self.pid_design_frame = ttk.Frame(self.notebook)
        self.create_pid_design_widgets(self.pid_design_frame)
        self.notebook.add(self.pid_design_frame, text="PID Design")
        self.general_chat_frame = GeneralChatFrame(self.notebook)
        self.notebook.add(self.general_chat_frame, text="General Chat")
        self.result_eval_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.result_eval_frame, text="Result Evaluation")
        self.feedback_frame = FeedbackFrame(self.notebook)
        self.notebook.add(self.feedback_frame, text="Feedback")
        self.project_manager_frame = ProjectManagerFrame(self.notebook, self)
        self.notebook.add(self.project_manager_frame, text="Projects")
        self.performance_comparison_frame = PerformanceComparisonFrame(self.notebook, self.performance_results)
        self.notebook.add(self.performance_comparison_frame, text="Performance Comparison")
        self.graph_preview_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.graph_preview_frame, text="Graph Preview")
    
    def create_pid_design_widgets(self, frame):
        lbl_title = ttk.Label(frame, text="Control Problem Description", font=("Helvetica", 16, "bold"))
        lbl_title.pack(pady=10)
        self.problem_text = tk.Text(frame, width=80, height=5, font=("Helvetica", 12))
        self.problem_text.insert(tk.END, "Example: I want a settling time shorter than 2.2 seconds, a maximum overshoot of 5%, and a rise time faster than 0.5 seconds for my system.")
        self.problem_text.pack(padx=10, pady=5, fill="both", expand=True)
        ToolTip(self.problem_text, "Enter your control problem description. For example, 'I want a settling time shorter than 2.2 seconds, a maximum overshoot of 5%, and a rise time faster than 0.5 seconds.'")
        
        tf_frame = ttk.LabelFrame(frame, text="Transfer Function Selection", style="TLabelframe")
        tf_frame.pack(padx=10, pady=5, fill="both", expand=True)
        
        self.tf_method = tk.StringVar(value="manual")
        method_frame = ttk.Frame(tf_frame)
        method_frame.pack(pady=5, fill="x")
        ttk.Label(method_frame, text="Transfer Function Method:", font=("Helvetica", 10, "bold")).pack(side="left", padx=5)
        ttk.Radiobutton(method_frame, text="Manual Entry", variable=self.tf_method, value="manual", command=self.update_tf_input, style="TRadiobutton").pack(side="left", padx=5)
        ttk.Radiobutton(method_frame, text="Sample Selection", variable=self.tf_method, value="sample", command=self.update_tf_input, style="TRadiobutton").pack(side="left", padx=5)
        
        self.manual_tf_frame = ttk.Frame(tf_frame)
        self.manual_tf_frame.pack(padx=10, pady=5, fill="both", expand=True)
        ttk.Label(self.manual_tf_frame, text="Numerator (comma separated):", font=("Helvetica", 10)).pack(side="left", padx=5)
        self.numerator_entry = ttk.Entry(self.manual_tf_frame, width=20)
        self.numerator_entry.pack(side="left", padx=5)
        ttk.Label(self.manual_tf_frame, text="Denominator (comma separated):", font=("Helvetica", 10)).pack(side="left", padx=5)
        self.denom_entry = ttk.Entry(self.manual_tf_frame, width=20)
        self.denom_entry.pack(side="left", padx=5)
        self.numerator_entry.insert(0, "1")
        self.denom_entry.insert(0, "1,2,1")
        self.numerator_entry.bind("<FocusOut>", lambda e: self.validate_manual_tf())
        self.denom_entry.bind("<FocusOut>", lambda e: self.validate_manual_tf())
        
        self.sample_tf_frame = ttk.Frame(tf_frame)
        self.sample_tf_frame.pack_forget()
        ttk.Label(self.sample_tf_frame, text="Sample Transfer Function:", font=("Helvetica", 10)).pack(side="left", padx=5)
        sample_names = [sample["name"] for sample in sample_transfer_functions]
        self.sample_tf_combobox = ttk.Combobox(self.sample_tf_frame, values=sample_names, state="readonly", width=50)
        self.sample_tf_combobox.pack(side="left", padx=5)
        self.sample_tf_combobox.current(0)
        self.sample_tf_combobox.bind("<<ComboboxSelected>>", lambda e: self.update_tf_formula())
        
        self.tf_formula_label = ttk.Label(tf_frame, text="Selected Transfer Function: ", font=("Helvetica", 10, "italic"))
        self.tf_formula_label.pack(padx=10, pady=5, fill="x")
        self.tf_desc_label = ttk.Label(tf_frame, text="Description: ", font=("Helvetica", 10, "italic"))
        self.tf_desc_label.pack(padx=10, pady=5, fill="x")
        self.tf_formula_image_label = ttk.Label(tf_frame, relief="flat", borderwidth=0)
        self.tf_formula_image_label.pack(padx=10, pady=5)
        self.update_tf_formula()
        
        btn_start = ttk.Button(frame, text="Start PID Design", command=self.open_pid_design_window)
        btn_start.pack(pady=10)
    
    def update_tf_input(self):
        if self.tf_method.get() == "manual":
            self.manual_tf_frame.pack(padx=10, pady=5, fill="both", expand=True)
            self.sample_tf_frame.pack_forget()
        else:
            self.sample_tf_frame.pack(padx=10, pady=5, fill="both", expand=True)
            self.manual_tf_frame.pack_forget()
        self.update_tf_formula()
    
    def update_tf_formula(self):
        if self.tf_method.get() == "manual":
            num_str = self.numerator_entry.get().strip()
            den_str = self.denom_entry.get().strip()
            formula_text = f"tf([{num_str}], [{den_str}])"
            desc_text = "Manual entry selected. Please verify your inputs."
            try:
                num_list = [float(x) for x in num_str.replace(" ", "").split(",") if x != ""]
                den_list = [float(x) for x in den_str.replace(" ", "").split(",") if x != ""]
            except:
                num_list, den_list = [1], [1,2,1]
            latex_formula = tf_to_latex(num_list, den_list)
        else:
            selected = self.sample_tf_combobox.get()
            sample = next((s for s in sample_transfer_functions if s["name"] == selected), None)
            if sample is not None:
                num = ", ".join(str(x) for x in sample["num"])
                den = ", ".join(str(x) for x in sample["den"])
                formula_text = f"tf([{num}], [{den}])"
                desc_text = sample["desc"]
                latex_formula = tf_to_latex(sample["num"], sample["den"])
            else:
                formula_text = "tf([1], [1,2,1])"
                desc_text = "Default"
                latex_formula = tf_to_latex([1], [1,2,1])
        self.tf_formula_label.config(text=f"Selected Transfer Function: {formula_text}")
        self.tf_desc_label.config(text=f"Description: {desc_text}")
        img = render_formula_image(latex_formula)
        if img:
            self.tf_formula_image_label.config(image=img)
            self.tf_formula_image_label.image = img
    
    def validate_manual_tf(self):
        num_str = self.numerator_entry.get().strip()
        den_str = self.denom_entry.get().strip()
        try:
            num_list = [float(x) for x in num_str.replace(" ", "").split(",") if x != ""]
            den_list = [float(x) for x in den_str.replace(" ", "").split(",") if x != ""]
            if len(den_list) == 0 or len(num_list) == 0:
                raise ValueError("List cannot be empty.")
        except Exception as e:
            messagebox.showerror("Error", f"Invalid transfer function input: {e}\nDefault values will be used.")
            self.numerator_entry.delete(0, tk.END)
            self.denom_entry.delete(0, tk.END)
            self.numerator_entry.insert(0, "1")
            self.denom_entry.insert(0, "1,2,1")
        self.update_tf_formula()
    
    def open_pid_design_window(self):
        description = self.problem_text.get("1.0", tk.END).strip()
        if not description:
            messagebox.showwarning("Warning", "Please enter the control problem description.")
            return
        target_settling_time = extract_target_settling_time(description)
        target_overshoot = extract_target_overshoot(description)
        target_rise_time = extract_target_rise_time(description)
        main_app = self.winfo_toplevel()
        main_app.target_settling_time = target_settling_time
        main_app.target_overshoot = target_overshoot
        main_app.target_rise_time = target_rise_time
        messagebox.showinfo("Target Values", 
            f"Target Settling Time: {target_settling_time} s\n"
            f"Target Maximum Overshoot: {target_overshoot}%\n"
            f"Target Minimum Rise Time: {target_rise_time} s")
        if self.tf_method.get() == "manual":
            num_str = self.numerator_entry.get().strip()
            den_str = self.denom_entry.get().strip()
            try:
                num = [float(x) for x in num_str.replace(" ", "").split(",") if x != ""]
                den = [float(x) for x in den_str.replace(" ", "").split(",") if x != ""]
            except Exception as e:
                messagebox.showwarning("Warning", f"Invalid transfer function input: {e}. Default values will be used.")
                num = [1]
                den = [1,2,1]
        else:
            selected = self.sample_tf_combobox.get()
            sample = next((s for s in sample_transfer_functions if s["name"] == selected), None)
            if sample is not None:
                num = sample["num"]
                den = sample["den"]
            else:
                num = [1]
                den = [1,2,1]
        try:
            plant = tf(num, den)
        except Exception as e:
            messagebox.showwarning("Warning", f"Could not create transfer function: {e}. Default will be used.")
            plant = tf([1], [1,2,1])
        InteractiveDesignWindow(self, plant, target_settling_time, self.on_design_complete,
                                  target_overshoot, target_rise_time)
    
    def on_design_complete(self, plant, pid_params, target_settling_time):
        ph_agent = PyHesapAgent()
        sim_time = max(5, 4 * target_settling_time)
        settling_time, t, y, closed_loop, rise_time, overshoot = ph_agent.simulate_response(plant, pid_params, simulation_time=sim_time)
        self.final_pid_params = pid_params
        self.final_settling_time = settling_time
        self.final_plant = plant
        result_dict = {
            "pid_params": pid_params,
            "target_settling_time": target_settling_time,
            "final_settling_time": settling_time,
            "rise_time": rise_time,
            "overshoot": overshoot,
            "t": t.tolist(),
            "y": y.tolist()
        }
        self.performance_results.append(result_dict)
        self.performance_comparison_frame.update_listbox()
        for widget in self.result_eval_frame.winfo_children():
            widget.destroy()
        result_frame = ResultEvaluationFrame(self.result_eval_frame, self.final_pid_params, settling_time, target_settling_time, self.final_plant)
        result_frame.pack(fill="both", expand=True)
        self.notebook.select(self.result_eval_frame)
        self.graph_preview_locked = True
    
    def update_graph_preview(self, photo):
        if self.graph_preview_locked:
            return
        for widget in self.graph_preview_frame.winfo_children():
            widget.destroy()
        label = ttk.Label(self.graph_preview_frame, image=photo)
        label.image = photo
        label.pack(padx=10, pady=10)
    
    def create_footer(self):
        footer_frame = ttk.Frame(self, style="TFrame")
        footer_frame.pack(side="bottom", fill="x", padx=10, pady=5)
        footer_label = ttk.Label(footer_frame, text="Developed by the CS Research Group", font=("Helvetica", 10, "italic"), foreground="#555555")
        footer_label.pack(side="right")
    
    def generate_result_figure(self):
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        ax.plot(self.t, self.y, 'b-', linewidth=2, label="System Response (Y(t))")
        ax.set_xlabel("Time (s)", fontsize=12)
        ax.set_ylabel("Output Y(t)", fontsize=12)
        ax.set_title("Final Closed-Loop Step Response", fontsize=14)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.minorticks_on()
        ax.axhline(y=1, color='k', linestyle=':', linewidth=1.5, label="Target Output (1)")
        settle_idx = None
        for i in range(len(self.y)):
            if all(0.98 <= self.y[j] <= 1.02 for j in range(i, len(self.y))):
                settle_idx = i
                break
        if settle_idx is not None:
            ax.plot(self.t[settle_idx], self.y[settle_idx], 'ro', markersize=8, label="Settling Point")
            ax.axvline(x=self.t[settle_idx], color='r', linestyle='--', linewidth=1)
        start_idx = next((i for i, val in enumerate(self.y) if val >= 0.1), None)
        end_idx = next((i for i, val in enumerate(self.y) if val >= 0.9), None)
        if start_idx is not None and end_idx is not None:
            ax.plot(self.t[start_idx], self.y[start_idx], 'go', markersize=8, label="Rise Start")
            ax.plot(self.t[end_idx], self.y[end_idx], 'mo', markersize=8, label="Rise End")
            ax.axvline(x=self.t[start_idx], color='g', linestyle='--', linewidth=1)
            ax.axvline(x=self.t[end_idx], color='m', linestyle='--', linewidth=1)
        max_idx = np.argmax(self.y)
        if self.y[max_idx] > 1:
            ax.plot(self.t[max_idx], self.y[max_idx], 'co', markersize=8, label="Maximum Overshoot Point")
            ax.axhline(y=self.y[max_idx], color='c', linestyle='--', linewidth=1)
        ax.axvline(x=self.master.target_settling_time, color='b', linestyle='--', linewidth=1.5, label="Target Settling Time")
        pid_controller = tf([self.pid_params['Kd'], self.pid_params['Kp'], self.pid_params['Ki']], [1, 0])
        L = series(pid_controller, tf([1], [1,2,1]))
        gm, pm, wg, wp = margin(L)
        rise_time_str = f"{self.optimized_metrics['rise_time']:.3f}" if self.optimized_metrics['rise_time'] is not None else "Not determined"
        annot_text = (
            f"Settling Time: {self.optimized_metrics['settling_time']:.3f} s\n"
            f"Rise Time: {rise_time_str} s\n"
            f"Max Overshoot: {self.optimized_metrics['overshoot']:.2f}%\n"
            f"Phase Margin: {pm:.2f}°\n"
            f"Gain Margin: {'Infinite' if gm==np.inf else f'{gm:.2f} dB'}"
        )
        ax.text(0.65, 0.80, annot_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7))
        ax.legend(loc="best")
        fig.tight_layout()
        return fig

    def show_results(self):
        fig = self.generate_result_figure()
        def on_click(event):
            if event.inaxes is not None:
                annotation_text = simpledialog.askstring("Add Label", "Enter label text:")
                if annotation_text:
                    ax = fig.get_axes()[0]
                    ax.annotate(annotation_text, xy=(event.xdata, event.ydata),
                                xytext=(event.xdata + 0.1, event.ydata + 0.1),
                                arrowprops=dict(facecolor='black', arrowstyle='->'))
                    canvas.draw()
        canvas = FigureCanvasTkAgg(fig, master=self.master)
        fig.canvas.mpl_connect("button_press_event", on_click)
        
        graph_window = tk.Toplevel(self)
        graph_window.title("Graph Preview")
        canvas = FigureCanvasTkAgg(fig, master=graph_window)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        toolbar = NavigationToolbar2Tk(canvas, graph_window)
        toolbar.update()
        canvas.draw()
        graph_window.protocol("WM_DELETE_WINDOW", lambda: self.on_graph_window_closed(fig, graph_window))
    
    def on_graph_window_closed(self, fig, graph_window):
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        self.master.saved_graph_bytes = buf.getvalue()
        photo = ImageTk.PhotoImage(Image.open(buf))
        self.master.saved_graph_photo = photo
        self.master.update_graph_preview(photo)
        graph_window.destroy()
    
    def accept_design(self):
        rise_time_str = f"{self.optimized_metrics['rise_time']:.3f}" if self.optimized_metrics['rise_time'] is not None else "N/A"
        msg = (
            f"Design accepted!\nFinal PID: {self.pid_params}\n"
            f"Final Settling Time: {self.optimized_metrics['settling_time']:.3f} s (Measured)\n"
            f"Target Settling Time: {self.master.target_settling_time:.3f} s (Initial)\n"
            f"Final Rise Time: {rise_time_str} s (Target: {self.target_rise_time} s)\n"
            f"Final Max Overshoot: {self.optimized_metrics['overshoot']:.2f}% (Target: {self.target_overshoot}%)"
        )
        messagebox.showinfo("Design Accepted", msg)
        fig = self.generate_result_figure()
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        photo = ImageTk.PhotoImage(Image.open(buf))
        self.master.saved_graph_bytes = buf.getvalue()
        self.master.saved_graph_photo = photo
        self.master.update_graph_preview(photo)
        self.show_results()
        self.on_design_complete(self.plant, self.pid_params, self.master.target_settling_time)
        self.destroy()

# ==============================
# Main Application Startup
# ==============================
if __name__ == "__main__":
    app = MainApplication()
    def updated_update_graph_preview(photo):
        if app.graph_preview_locked:
            return
        for widget in app.graph_preview_frame.winfo_children():
            widget.destroy()
        label = ttk.Label(app.graph_preview_frame, image=photo)
        label.image = photo
        label.pack(padx=10, pady=10)
    app.update_graph_preview = updated_update_graph_preview
    app.mainloop()
    backup_code()
