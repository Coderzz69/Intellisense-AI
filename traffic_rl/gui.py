import tkinter as tk
from tkinter import ttk, scrolledtext
import subprocess
import threading
import queue

class TrafficDemoGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("SSS — Intelligent Traffic Management System")
        self.root.geometry("880x720")
        self.root.configure(bg="#1E1E2E")
        
        # Style Configuration
        style = ttk.Style()
        style.theme_use("clam")
        
        style.configure("TFrame", background="#1E1E2E")
        style.configure("TLabel", background="#1E1E2E", foreground="#D9E0EE", font=("Segoe UI", 12))
        style.configure("Header.TLabel", font=("Segoe UI", 18, "bold"), foreground="#F28FAD")
        
        style.configure("TRadiobutton", background="#1E1E2E", foreground="#D9E0EE", font=("Segoe UI", 11))
        style.map("TRadiobutton",
                  background=[("active", "#1E1E2E")],
                  indicatorcolor=[("selected", "#F28FAD")])
        
        style.configure("Run.TButton", font=("Segoe UI", 14, "bold"), 
                        background="#ABE9B3", foreground="#1E1E2E", padding=10)
        style.map("Run.TButton", background=[("active", "#8BD5CA")])

        # Title
        ttk.Label(root, text="SSS Smart Traffic Control Panel",
                  style="Header.TLabel").pack(pady=20)

        # ─── Controls Frame ───
        control_frame = ttk.Frame(root)
        control_frame.pack(fill="x", padx=40, pady=10)
        
        # Scenarios List
        scenario_frame = tk.LabelFrame(control_frame, text=" Select Traffic Scenario ", 
                                     bg="#1E1E2E", fg="#F28FAD", font=("Segoe UI", 11, "bold"), padx=20, pady=10)
        scenario_frame.pack(side="left", fill="both", expand=True, padx=10)
        
        self.scenario_var = tk.StringVar(value="normal")
        ttk.Radiobutton(scenario_frame, text="Normal Traffic (Volume Based)", variable=self.scenario_var, value="normal").pack(anchor="w", pady=5)
        ttk.Radiobutton(scenario_frame, text="Heavy Congestion (Erratic Indian Traffic)", variable=self.scenario_var, value="heavy").pack(anchor="w", pady=5)
        ttk.Radiobutton(scenario_frame, text="Emergency Interception (Ambulance Spawning)", variable=self.scenario_var, value="emergency").pack(anchor="w", pady=5)

        # Mode List
        mode_frame = tk.LabelFrame(control_frame, text=" Select Controller Profile ", 
                                   bg="#1E1E2E", fg="#F28FAD", font=("Segoe UI", 11, "bold"), padx=20, pady=10)
        mode_frame.pack(side="right", fill="both", expand=True, padx=10)
        
        self.mode_var = tk.StringVar(value="triple")
        ttk.Radiobutton(mode_frame, text="[TRIPLE]  Baseline vs Hybrid AI vs Master AI",
                        variable=self.mode_var, value="triple").pack(anchor="w", pady=5)
        ttk.Radiobutton(mode_frame, text="[COMPARE]  Baseline vs Master AI",
                        variable=self.mode_var, value="compare").pack(anchor="w", pady=5)
        ttk.Radiobutton(mode_frame, text="[HYBRID]  Baseline vs Hybrid AI",
                        variable=self.mode_var, value="hybrid").pack(anchor="w", pady=5)
        ttk.Radiobutton(mode_frame, text="[AI]  Master Hierarchical AI only",
                        variable=self.mode_var, value="ai").pack(anchor="w", pady=5)
        ttk.Radiobutton(mode_frame, text="[BASELINE]  Fixed-Timer Baseline only",
                        variable=self.mode_var, value="baseline").pack(anchor="w", pady=5)

        # Headless checkbox
        self.headless_var = tk.BooleanVar(value=False)
        chk = tk.Checkbutton(root, text="Headless Mode (Do not open SUMO Visualizer)", variable=self.headless_var,
                             bg="#1E1E2E", fg="#D9E0EE", selectcolor="#282A36", font=("Segoe UI", 10))
        chk.pack(pady=5)

        # ─── Launch Button ───
        self.run_btn = ttk.Button(root, text="▶ LAUNCH SIMULATION", style="Run.TButton", command=self.run_simulation)
        self.run_btn.pack(pady=15)

        # ─── Output Log ───
        ttk.Label(root, text="Live Output & Metrics:").pack(anchor="w", padx=40)
        self.log_area = scrolledtext.ScrolledText(root, height=12, bg="#282A36", fg="#F8F8F2", 
                                                font=("Consolas", 10), bd=0)
        self.log_area.pack(fill="both", expand=True, padx=40, pady=(5, 20))
        
        self.queue = queue.Queue()
        self.process = None

    def log(self, text):
        self.log_area.insert(tk.END, text + "\n")
        self.log_area.see(tk.END)

    def read_process(self):
        # Read subprocess output natively in a background thread
        for line in iter(self.process.stdout.readline, b''):
            decoded_line = line.decode("utf-8", errors="replace").strip()
            # Clean up ANSI color codes so they don't look weird in tkinter
            import re
            clean_line = re.sub(r'\x1b\[.*?m', '', decoded_line)
            self.queue.put(clean_line)
            
        self.process.stdout.close()
        self.process.wait()
        self.queue.put("=== SIMULATION COMPLETED ===")

    def update_log(self):
        try:
            while True:
                line = self.queue.get_nowait()
                self.log(line)
                if line == "=== SIMULATION COMPLETED ===":
                    self.run_btn.config(state="normal")
                    self.run_btn.config(text="▶ LAUNCH SIMULATION")
        except queue.Empty:
            pass
        self.root.after(100, self.update_log)

    def run_simulation(self):
        scenario = self.scenario_var.get()
        mode = self.mode_var.get()
        
        cmd = ["python", "demo.py", "--scenario", scenario, "--mode", mode]
        if self.headless_var.get():
            cmd.append("--headless")

        self.log_area.delete("1.0", tk.END)
        self.log(f"Executing: {' '.join(cmd)}")
        self.log("Initializing TraCI Environment...\n" + "-"*60)
        
        self.run_btn.config(state="disabled")
        self.run_btn.config(text="Running...")
        
        # Start subprocess
        self.process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        
        # Stream metrics securely in background
        threading.Thread(target=self.read_process, daemon=True).start()
        self.update_log()

if __name__ == "__main__":
    root = tk.Tk()
    app = TrafficDemoGUI(root)
    root.mainloop()
