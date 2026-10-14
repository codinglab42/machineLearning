import subprocess
import re
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configurazione stile grafico
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.sans-serif': 'DejaVu Sans', 'font.size': 10})

def run_executable(exec_path):
    """Esegue un binario C++ e restituisce l'output della console."""
    path = Path(exec_path)
    if not path.exists():
        print(f"⚠️ Warning: Impossibile trovare il binario in {exec_path}")
        return None
    
    print(f"🚀 Esecuzione benchmark C++: {exec_path}...")
    res = subprocess.run([str(path)], capture_output=True, text=True, check=True)
    return res.stdout

# ==========================================
# PARSERS PER I TRE BENCHMARK
# ==========================================

def parse_linear_output(stdout):
    if not stdout: return None
    data = {'sample_scaling': [], 'feature_scaling': [], 'solvers': [], 'regularization': []}
    
    # Sample scaling
    m = re.search(r"BENCHMARK 1: Scaling con numero di campioni\n.*?\n-+\n(.*?)\n\n", stdout, re.DOTALL)
    if m:
        for line in m.group(1).strip().split('\n'):
            p = line.split()
            if len(p) >= 3:
                data['sample_scaling'].append({'samples': int(p[0]), 'time_ms': float(p[1]), 'r2': float(p[2])})

    # Feature scaling
    m = re.search(r"BENCHMARK 2: Scaling con numero di feature\n.*?\n-+\n(.*?)\n\n", stdout, re.DOTALL)
    if m:
        for line in m.group(1).strip().split('\n'):
            p = line.split()
            if len(p) >= 3:
                data['feature_scaling'].append({'features': int(p[0]), 'time_ms': float(p[1]), 'r2': float(p[2])})

    # Solvers
    m = re.search(r"BENCHMARK 3: Confronto tra solver\n.*?\n-+\n(.*?)\n\n", stdout, re.DOTALL)
    if m:
        for line in m.group(1).strip().split('\n'):
            p = line.rsplit(None, 2)
            if len(p) == 3:
                data['solvers'].append({'solver': p[0].strip(), 'time_ms': float(p[1]), 'r2': float(p[2])})

    # Regularization
    m = re.search(r"BENCHMARK 4: Impatto della regolarizzazione\n.*?\n-+\n(.*?)\n\n", stdout, re.DOTALL)
    if m:
        for line in m.group(1).strip().split('\n'):
            p = line.split()
            if len(p) >= 3:
                data['regularization'].append({'lambda': float(p[0]), 'time_ms': float(p[1]), 'r2': float(p[2])})

    return data

def parse_logistic_output(stdout):
    if not stdout: return None
    data = {'sample_scaling': [], 'feature_scaling': [], 'regularization': []}

    # Sample scaling
    m = re.search(r"LOGISTIC REGRESSION - Scaling con numero di campioni\n.*?\n-+\n(.*?)\n\n", stdout, re.DOTALL)
    if m:
        for line in m.group(1).strip().split('\n'):
            p = line.split()
            if len(p) >= 4:
                data['sample_scaling'].append({'samples': int(p[0]), 'time_ms': float(p[1]), 'accuracy': float(p[2]), 'iters': int(p[3])})

    # Feature scaling
    m = re.search(r"LOGISTIC REGRESSION - Scaling con numero di feature\n.*?\n-+\n(.*?)\n\n", stdout, re.DOTALL)
    if m:
        for line in m.group(1).strip().split('\n'):
            p = line.split()
            if len(p) >= 4:
                data['feature_scaling'].append({'features': int(p[0]), 'time_ms': float(p[1]), 'accuracy': float(p[2]), 'iters': int(p[3])})

    # Regularization
    m = re.search(r"LOGISTIC REGRESSION - Impatto della regolarizzazione\n.*?\n-+\n(.*?)\n\n", stdout, re.DOTALL)
    if m:
        for line in m.group(1).strip().split('\n'):
            p = line.split()
            if len(p) >= 4:
                data['regularization'].append({'lambda': float(p[0]), 'time_ms': float(p[1]), 'accuracy': float(p[2]), 'iters': int(p[3])})

    return data

def parse_neural_output(stdout):
    if not stdout: return None
    data = {'sample_scaling': [], 'feature_scaling': [], 'neuron_scaling': [], 'optimizers': []}

    # Sample scaling
    m = re.search(r"NEURAL NETWORK - Scaling con numero di campioni\n.*?\n-+\n(.*?)\n\n", stdout, re.DOTALL)
    if m:
        for line in m.group(1).strip().split('\n'):
            p = line.split()
            if len(p) >= 4:
                data['sample_scaling'].append({'samples': int(p[0]), 'time_ms': float(p[1]), 'accuracy': float(p[2]), 'loss': float(p[3])})

    # Feature scaling
    m = re.search(r"NEURAL NETWORK - Scaling con numero di feature\n.*?\n-+\n(.*?)\n\n", stdout, re.DOTALL)
    if m:
        for line in m.group(1).strip().split('\n'):
            p = line.split()
            if len(p) >= 4:
                data['feature_scaling'].append({'features': int(p[0]), 'time_ms': float(p[1]), 'accuracy': float(p[2]), 'loss': float(p[3])})

    # Neuron scaling
    m = re.search(r"NEURAL NETWORK - Scaling con numero di neuroni\n.*?\n-+\n(.*?)\n\n", stdout, re.DOTALL)
    if m:
        for line in m.group(1).strip().split('\n'):
            p = line.split()
            if len(p) >= 4:
                data['neuron_scaling'].append({'arch': p[0], 'time_ms': float(p[1]), 'accuracy': float(p[2]), 'params': int(p[3])})

    # Optimizers
    m = re.search(r"NEURAL NETWORK - Confronto ottimizzatori\n.*?\n-+\n(.*?)\n\n", stdout, re.DOTALL)
    if m:
        for line in m.group(1).strip().split('\n'):
            p = line.split()
            if len(p) >= 4:
                data['optimizers'].append({'optimizer': p[0], 'time_ms': float(p[1]), 'accuracy': float(p[2]), 'loss': float(p[3])})

    return data

# ==========================================
# PLOTTING FUNCTIONS
# ==========================================

def plot_linear_summary(data, out_dir):
    if not data: return
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("Linear Regression Benchmark Results", fontsize=14, fontweight='bold')

    # 1. Sample Scaling
    if data['sample_scaling']:
        x = [d['samples'] for d in data['sample_scaling']]
        y = [d['time_ms'] for d in data['sample_scaling']]
        axes[0,0].plot(x, y, marker='o', color='#2ca02c', linewidth=2)
        axes[0,0].set_title("Execution Time vs Sample Size")
        axes[0,0].set_xlabel("Samples")
        axes[0,0].set_ylabel("Time (ms)")

    # 2. Feature Scaling
    if data['feature_scaling']:
        x = [d['features'] for d in data['feature_scaling']]
        y = [d['time_ms'] for d in data['feature_scaling']]
        axes[0,1].plot(x, y, marker='s', color='#1f77b4', linewidth=2)
        axes[0,1].set_title("Execution Time vs Feature Count")
        axes[0,1].set_xlabel("Features")
        axes[0,1].set_ylabel("Time (ms)")

    # 3. Solver Comparison
    if data['solvers']:
        solvers = [d['solver'] for d in data['solvers']]
        times = [d['time_ms'] for d in data['solvers']]
        bars = axes[1,0].bar(solvers, times, color='#ff7f0e', alpha=0.85)
        axes[1,0].set_title("Solver Comparison (Execution Time)")
        axes[1,0].set_ylabel("Time (ms)")
        for b in bars:
            yval = b.get_height()
            axes[1,0].text(b.get_x() + b.get_width()/2.0, yval + 0.1, f"{yval:.1f} ms", ha='center', va='bottom')

    # 4. Regularization
    if data['regularization']:
        x = [str(d['lambda']) for d in data['regularization']]
        y = [d['r2'] for d in data['regularization']]
        axes[1,1].plot(x, y, marker='^', color='#d62728', linewidth=2)
        axes[1,1].set_title("R² Score vs Lambda (Regularization)")
        axes[1,1].set_xlabel("Lambda")
        axes[1,1].set_ylabel("R² Score")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_dir / "benchmark_linear_results.png", dpi=300)
    print("📊 Grafico salvato: docs/assets/benchmark_linear_results.png")

def plot_logistic_summary(data, out_dir):
    if not data: return
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle("Logistic Regression Benchmark Results", fontsize=14, fontweight='bold')

    # 1. Sample Scaling
    if data['sample_scaling']:
        x = [d['samples'] for d in data['sample_scaling']]
        y = [d['time_ms'] for d in data['sample_scaling']]
        axes[0].plot(x, y, marker='o', color='#1f77b4', linewidth=2)
        axes[0].set_title("Execution Time vs Sample Size")
        axes[0].set_xlabel("Samples")
        axes[0].set_ylabel("Time (ms)")

    # 2. Feature Scaling
    if data['feature_scaling']:
        x = [d['features'] for d in data['feature_scaling']]
        y = [d['time_ms'] for d in data['feature_scaling']]
        axes[1].plot(x, y, marker='s', color='#2ca02c', linewidth=2)
        axes[1].set_title("Execution Time vs Feature Count")
        axes[1].set_xlabel("Features")
        axes[1].set_ylabel("Time (ms)")

    # 3. Regularization & Accuracy
    if data['regularization']:
        x = [str(d['lambda']) for d in data['regularization']]
        acc = [d['accuracy'] * 100 for d in data['regularization']]
        axes[2].plot(x, acc, marker='d', color='#9467bd', linewidth=2)
        axes[2].set_title("Accuracy (%) vs Lambda")
        axes[2].set_xlabel("Lambda")
        axes[2].set_ylabel("Accuracy (%)")

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(out_dir / "benchmark_logistic_results.png", dpi=300)
    print("📊 Grafico salvato: docs/assets/benchmark_logistic_results.png")

def plot_neural_summary(data, out_dir):
    if not data: return
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("Neural Network Benchmark Results", fontsize=14, fontweight='bold')

    # 1. Throughput
    if data['sample_scaling']:
        samples = [d['samples'] for d in data['sample_scaling']]
        throughput = [d['samples'] / (d['time_ms'] / 1000.0) if d['time_ms'] > 0 else 0 for d in data['sample_scaling']]
        axes[0,0].plot(samples, throughput, marker='o', color='#1f77b4', linewidth=2)
        axes[0,0].set_title("Training Throughput (Samples/sec)")
        axes[0,0].set_xlabel("Samples")
        axes[0,0].set_ylabel("Samples / sec")

    # 2. Feature Scaling
    if data['feature_scaling']:
        x = [d['features'] for d in data['feature_scaling']]
        y = [d['time_ms'] for d in data['feature_scaling']]
        axes[0,1].plot(x, y, marker='s', color='#2ca02c', linewidth=2)
        axes[0,1].set_title("Execution Time vs Feature Count")
        axes[0,1].set_xlabel("Features")
        axes[0,1].set_ylabel("Time (ms)")

    # 3. Architecture Scaling
    if data['neuron_scaling']:
        archs = [d['arch'] for d in data['neuron_scaling']]
        times = [d['time_ms'] for d in data['neuron_scaling']]
        bars = axes[1,0].bar(archs, times, color='#ff7f0e', alpha=0.85)
        axes[1,0].set_title("Execution Time by Architecture")
        axes[1,0].set_xlabel("Architecture")
        axes[1,0].set_ylabel("Time (ms)")

    # 4. Optimizers
    if data['optimizers']:
        opts = [d['optimizer'] for d in data['optimizers']]
        accs = [d['accuracy'] * 100 for d in data['optimizers']]
        bars = axes[1,1].bar(opts, accs, color=['#d62728', '#9467bd', '#8c564b'], alpha=0.85)
        axes[1,1].set_title("Optimizer Convergence (Accuracy %)")
        axes[1,1].set_ylabel("Accuracy (%)")
        axes[1,1].set_ylim(0, 105)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_dir / "benchmark_neural_results.png", dpi=300)
    print("📊 Grafico salvato: docs/assets/benchmark_neural_results.png")


# ==========================================
# MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    output_dir = Path("docs/assets")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Definizione percorsi dei binari compilati
    exec_linear = "./build/tests/performance/benchmark_linear"
    exec_logistic = "./build/tests/performance/benchmark_logistic"
    exec_neural = "./build/tests/performance/benchmark_neural"

    # Linear
    out_lin = run_executable(exec_linear)
    data_lin = parse_linear_output(out_lin)
    plot_linear_summary(data_lin, output_dir)

    # Logistic
    out_log = run_executable(exec_logistic)
    data_log = parse_logistic_output(out_log)
    plot_logistic_summary(data_log, output_dir)

    # Neural
    out_neu = run_executable(exec_neural)
    data_neu = parse_neural_output(out_neu)
    plot_neural_summary(data_neu, output_dir)

    print("\n✅ Suite di benchmark completata! Tutti i grafici sono stati aggiornati.")