import argparse
import csv
import hashlib
import json
import math
import os
import platform
import re
import statistics
import subprocess
from datetime import datetime, timezone
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configurazione stile grafico
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.sans-serif': 'DejaVu Sans', 'font.size': 10})

def run_executable(exec_path):
    """Esegue un binario C++ e restituisce l'output della console."""
    path = Path(exec_path)
    if not path.is_file():
        raise FileNotFoundError(f"Benchmark executable not found: {exec_path}")
    print(f"🚀 Esecuzione benchmark C++: {exec_path}...")
    res = subprocess.run([str(path)], capture_output=True, text=True)
    if res.returncode:
        raise RuntimeError(f"Benchmark {path} failed with exit code {res.returncode}:\n{res.stdout}\n{res.stderr}")
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
                data['sample_scaling'].append({'samples': int(p[0]), 'time_ms': float(p[1]), 'r2': float(p[2]), 'iterations': int(p[3])})

    # Feature scaling
    m = re.search(r"BENCHMARK 2: Scaling con numero di feature\n.*?\n-+\n(.*?)\n\n", stdout, re.DOTALL)
    if m:
        for line in m.group(1).strip().split('\n'):
            p = line.split()
            if len(p) >= 3:
                data['feature_scaling'].append({'features': int(p[0]), 'time_ms': float(p[1]), 'r2': float(p[2]), 'iterations': int(p[3])})

    # Solvers
    m = re.search(r"BENCHMARK 3: Confronto tra solver\n.*?\n-+\n(.*?)\n\n", stdout, re.DOTALL)
    if m:
        for line in m.group(1).strip().split('\n'):
            p = line.rsplit(None, 3)
            if len(p) == 4:
                data['solvers'].append({'solver': p[0].strip(), 'time_ms': float(p[1]), 'r2': float(p[2]), 'iterations': int(p[3])})

    # Regularization
    m = re.search(r"BENCHMARK 4: Impatto della regolarizzazione\n.*?\n-+\n(.*?)\n\n", stdout, re.DOTALL)
    if m:
        for line in m.group(1).strip().split('\n'):
            p = line.split()
            if len(p) >= 3:
                data['regularization'].append({'lambda': float(p[0]), 'time_ms': float(p[1]), 'r2': float(p[2]), 'iterations': int(p[3])})

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
    data = {'sample_scaling': [], 'feature_scaling': [], 'neuron_scaling': [], 'optimizers': [], 'regularizers': []}

    # Sample scaling
    m = re.search(r"NEURAL NETWORK - Scaling con numero di campioni\n.*?\n-+\n(.*?)\n\n", stdout, re.DOTALL)
    if m:
        for line in m.group(1).strip().split('\n'):
            p = line.split()
            if len(p) == 5:
                data['sample_scaling'].append({'samples': int(p[0]), 'time_ms': float(p[1]), 'accuracy': float(p[2]), 'loss': float(p[3]), 'baseline': float(p[4])})

    # Feature scaling
    m = re.search(r"NEURAL NETWORK - Scaling con numero di feature\n.*?\n-+\n(.*?)\n\n", stdout, re.DOTALL)
    if m:
        for line in m.group(1).strip().split('\n'):
            p = line.split()
            if len(p) == 5:
                data['feature_scaling'].append({'features': int(p[0]), 'time_ms': float(p[1]), 'accuracy': float(p[2]), 'loss': float(p[3]), 'baseline': float(p[4])})

    # Neuron scaling
    m = re.search(r"NEURAL NETWORK - Scaling con numero di neuroni\n.*?\n-+\n(.*?)\n\n", stdout, re.DOTALL)
    if m:
        for line in m.group(1).strip().split('\n'):
            p = line.split()
            if len(p) == 5:
                data['neuron_scaling'].append({'arch': p[0], 'time_ms': float(p[1]), 'accuracy': float(p[2]), 'params': int(p[3]), 'baseline': float(p[4])})

    # Optimizers
    m = re.search(r"NEURAL NETWORK - Confronto ottimizzatori\n.*?\n-+\n(.*?)\n\n", stdout, re.DOTALL)
    if m:
        for line in m.group(1).strip().split('\n'):
            p = line.split()
            if len(p) == 5:
                data['optimizers'].append({'optimizer': p[0], 'time_ms': float(p[1]), 'accuracy': float(p[2]), 'loss': float(p[3]), 'baseline': float(p[4])})

    m = re.search(r"NEURAL NETWORK - Confronto regolarizzatori\n.*?\n-+\n(.*?)\n\n", stdout, re.DOTALL)
    if m:
        for line in m.group(1).strip().split('\n'):
            p = line.split()
            if len(p) == 5:
                data['regularizers'].append({'regularizer': p[0], 'time_ms': float(p[1]), 'accuracy': float(p[2]), 'loss': float(p[3]), 'baseline': float(p[4])})

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
    plt.close(fig)
    print(f"📊 Grafico salvato: {out_dir / 'benchmark_linear_results.png'}")

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
    plt.close(fig)
    print(f"📊 Grafico salvato: {out_dir / 'benchmark_logistic_results.png'}")

def plot_neural_summary(data, out_dir):
    if not data: return
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("Neural Network Benchmark Results", fontsize=14, fontweight='bold')

    # 1. Training time
    if data['sample_scaling']:
        samples = [d['samples'] for d in data['sample_scaling']]
        axes[0,0].plot(samples, [d['time_ms'] for d in data['sample_scaling']], marker='o', color='#1f77b4', linewidth=2)
        axes[0,0].set_title("Training time vs dataset size")
        axes[0,0].set_xlabel("Total samples (80% training)")
        axes[0,0].set_ylabel("Time (ms)")

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
        axes[1,0].tick_params(axis='x', labelrotation=20)
        for label in axes[1,0].get_xticklabels():
            label.set_horizontalalignment('right')

    # 4. Optimizers
    if data['optimizers']:
        opts = [d['optimizer'] for d in data['optimizers']]
        accs = [d['accuracy'] * 100 for d in data['optimizers']]
        axes[1,1].bar(opts, accs, color='#9467bd', alpha=0.85)
        axes[1,1].axhline(data['optimizers'][0]['baseline'] * 100, color='black', linestyle='--', label='Majority baseline')
        axes[1,1].legend()
        axes[1,1].set_title("Optimizer test accuracy")
        axes[1,1].set_ylabel("Accuracy (%)")
        axes[1,1].set_ylim(0, 105)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_dir / "benchmark_neural_results.png", dpi=300)
    plt.close(fig)
    print(f"📊 Grafico salvato: {out_dir / 'benchmark_neural_results.png'}")


def parse_sequence_output(stdout):
    data = {'recurrent': [], 'checkpoint': []}
    for title, section in [('Recurrent training', 'recurrent'), ('Checkpoint in memory', 'checkpoint')]:
        match = re.search(rf"SEQUENCE - {title}\n-+\n[^\n]+\n(.*?)\n\n", stdout, re.DOTALL)
        if not match:
            raise ValueError(f"Missing sequence section: {title}")
        for line in match.group(1).splitlines():
            parts = line.split()
            if section == 'recurrent' and len(parts) == 4:
                data[section].append({'type': parts[0], 'time_ms': float(parts[1]), 'test_mse': float(parts[2]), 'baseline_mse': float(parts[3])})
            elif section == 'checkpoint' and len(parts) == 3:
                data[section].append({'operation': parts[0], 'time_ms': float(parts[1]), 'bytes': int(parts[2])})
            else:
                raise ValueError(f"Invalid sequence benchmark row: {line}")
    return data


EXPECTED = {
    'linear': {'sample_scaling': 5, 'feature_scaling': 5, 'solvers': 3, 'regularization': 5},
    'logistic': {'sample_scaling': 5, 'feature_scaling': 5, 'regularization': 5},
    'neural': {'sample_scaling': 5, 'feature_scaling': 5, 'neuron_scaling': 5, 'optimizers': 5, 'regularizers': 4},
    'sequences': {'recurrent': 3, 'checkpoint': 2},
}
PARSERS = {'linear': parse_linear_output, 'logistic': parse_logistic_output,
           'neural': parse_neural_output, 'sequences': parse_sequence_output}
CASE_KEYS = ('samples', 'features', 'solver', 'lambda', 'arch', 'optimizer',
             'regularizer', 'type', 'operation')


def validate_result(name, data):
    if not data or set(data) != set(EXPECTED[name]):
        raise ValueError(f"Incomplete output for {name}")
    for section, count in EXPECTED[name].items():
        if len(data[section]) != count:
            raise ValueError(f"{name}/{section}: expected {count} rows, got {len(data[section])}")
        for row in data[section]:
            if row['time_ms'] <= 0 or any(not math.isfinite(v) for v in row.values() if isinstance(v, (int, float))):
                raise ValueError(f"{name}/{section}: invalid measurement: {row}")
            if 'accuracy' in row and not 0 <= row['accuracy'] <= 1:
                raise ValueError(f"{name}/{section}: invalid accuracy: {row}")
            if 'baseline' in row and not 0.5 <= row['baseline'] <= 0.8:
                raise ValueError(f"{name}/{section}: unbalanced test labels: {row}")


def aggregate(runs):
    result = {}
    for name in EXPECTED:
        result[name] = {}
        for section in EXPECTED[name]:
            combined = []
            for index, first in enumerate(runs[0][name][section]):
                row = first.copy()
                for run in runs[1:]:
                    other = run[name][section][index]
                    if set(other) != set(first) or any(other.get(key) != first.get(key) for key in CASE_KEYS):
                        raise ValueError(f"Inconsistent benchmark rows: {name}/{section}/{index}")
                times = [run[name][section][index]['time_ms'] for run in runs]
                row['time_ms'] = statistics.median(times)
                row['min_ms'] = min(times)
                row['max_ms'] = max(times)
                for key, value in first.items():
                    if key != 'time_ms' and isinstance(value, (int, float)):
                        row[key] = statistics.median(run[name][section][index][key] for run in runs)
                combined.append(row)
            result[name][section] = combined
    return result


def save_csv(path, runs, summary):
    fields = ['model', 'section', 'case', 'run', 'time_ms', 'min_ms', 'max_ms', 'metrics']
    with path.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for run_number, data in enumerate(runs, 1):
            for model, sections in data.items():
                for section, rows in sections.items():
                    for row in rows:
                        case = next(value for key, value in row.items() if key != 'time_ms')
                        writer.writerow({'model': model, 'section': section, 'case': case, 'run': run_number,
                                         'time_ms': row['time_ms'], 'metrics': json.dumps({k: v for k, v in row.items() if k != 'time_ms'})})
        for model, sections in summary.items():
            for section, rows in sections.items():
                for row in rows:
                    case = next(value for key, value in row.items() if key not in ('time_ms', 'min_ms', 'max_ms'))
                    writer.writerow({'model': model, 'section': section, 'case': case, 'run': 'median',
                                     'time_ms': row['time_ms'], 'min_ms': row['min_ms'], 'max_ms': row['max_ms'],
                                     'metrics': json.dumps({k: v for k, v in row.items() if k not in ('time_ms', 'min_ms', 'max_ms')})})


def plot_sequence_summary(data, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    recurrent = data['recurrent']
    axes[0].bar([r['type'] for r in recurrent], [r['time_ms'] for r in recurrent])
    axes[0].set(title='Sequence training: median time', ylabel='Time (ms)')
    axes[1].bar([r['type'] for r in recurrent], [r['test_mse'] for r in recurrent])
    axes[1].axhline(recurrent[0]['baseline_mse'], color='black', linestyle='--', label='Zero baseline')
    axes[1].set(title='Sequence test MSE', ylabel='MSE')
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out_dir / 'benchmark_sequences_results.png', dpi=300)
    plt.close(fig)


def compare_baseline(current, baseline, max_regression):
    regressions = []
    for model, sections in current.items():
        for section, rows in sections.items():
            old_rows = baseline[model][section]
            if len(old_rows) != len(rows):
                raise ValueError(f'Baseline row count differs: {model}/{section}')
            for row, old in zip(rows, old_rows):
                for key in CASE_KEYS:
                    if row.get(key) != old.get(key):
                        raise ValueError(f'Baseline case differs: {model}/{section}/{key}')
                if old['time_ms'] <= 0:
                    raise ValueError(f'Invalid baseline time: {model}/{section}')
                percent = 100 * (row['time_ms'] / old['time_ms'] - 1)
                if percent > max_regression and not math.isclose(percent, max_regression, rel_tol=1e-12, abs_tol=1e-12):
                    case = next(value for key, value in row.items()
                                if key not in ('time_ms', 'min_ms', 'max_ms'))
                    regressions.append((model, section, case, percent))
    return regressions


def main():
    parser = argparse.ArgumentParser(description='Run benchmarks sequentially and save measurements and plots')
    parser.add_argument('--build-dir', type=Path, default=Path('build/benchmarks'))
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--repetitions', type=int, default=3)
    parser.add_argument('--baseline', type=Path, help='Previous results.json from a comparable environment and workload')
    parser.add_argument('--max-regression-pct', type=float, default=20.0)
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error('--repetitions must be positive')
    if not math.isfinite(args.max_regression_pct) or args.max_regression_pct < 0:
        parser.error('--max-regression-pct must be a non-negative finite number')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    runs = []
    for iteration in range(1, args.repetitions + 1):
        run = {}
        for name, parse in PARSERS.items():
            stdout = run_executable(args.build_dir.resolve() / 'bin' / f'benchmark_{name}')
            (args.output_dir / f'{name}_run_{iteration}.txt').write_text(stdout, encoding='utf-8')
            data = parse(stdout)
            validate_result(name, data)
            run[name] = data
        runs.append(run)
    summary = aggregate(runs)
    save_csv(args.output_dir / 'results.csv', runs, summary)
    (args.output_dir / 'results.json').write_text(json.dumps({'runs': runs, 'summary': summary}, indent=2) + '\n', encoding='utf-8')
    cache = (args.build_dir / 'CMakeCache.txt').read_text(encoding='utf-8')
    cache_values = {}
    for key in ('CMAKE_BUILD_TYPE', 'CMAKE_CXX_COMPILER', 'CMAKE_CXX_FLAGS', 'CMAKE_CXX_FLAGS_RELEASE', 'USE_OPENMP'):
        match = re.search(rf'^{key}:[^=]+=(.*)$', cache, re.MULTILINE)
        if match:
            cache_values[key] = match.group(1)
    cpuinfo = Path('/proc/cpuinfo')
    cpu_match = re.search(r'^model name\s*:\s*(.*)$', cpuinfo.read_text(), re.MULTILINE) if cpuinfo.exists() else None
    compiler = cache_values.get('CMAKE_CXX_COMPILER')
    if not compiler:
        raise ValueError(f'C++ compiler missing in {args.build_dir / "CMakeCache.txt"}')
    compiler_version = subprocess.run([compiler, '--version'], check=True, text=True, capture_output=True).stdout.splitlines()[0]
    metadata = {'timestamp_utc': datetime.now(timezone.utc).isoformat(),
                'git_revision': subprocess.run(['git', 'rev-parse', 'HEAD'], check=True, text=True, capture_output=True).stdout.strip(),
                'git_status': subprocess.run(['git', 'status', '--short'], check=True, text=True, capture_output=True).stdout.splitlines(),
                'platform': platform.platform(), 'processor': cpu_match.group(1) if cpu_match else platform.processor(),
                'available_cpus': len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count(),
                'repetitions': args.repetitions, 'build_dir': str(args.build_dir.resolve()),
                'cmake': cache_values, 'compiler_version': compiler_version,
                'executables_sha256': {name: hashlib.sha256((args.build_dir / 'bin' / f'benchmark_{name}').read_bytes()).hexdigest() for name in PARSERS},
                'timing': 'Median/min/max of fit wall-clock durations, except in-memory checkpoint save/load.',
                'note': 'Model quality uses held-out synthetic data; neural loss is final training loss.'}
    (args.output_dir / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    plot_linear_summary(summary['linear'], args.output_dir)
    plot_logistic_summary(summary['logistic'], args.output_dir)
    plot_neural_summary(summary['neural'], args.output_dir)
    plot_sequence_summary(summary['sequences'], args.output_dir)
    if args.baseline:
        baseline = json.loads(args.baseline.read_text(encoding='utf-8'))['summary']
        regressions = compare_baseline(summary, baseline, args.max_regression_pct)
        (args.output_dir / 'regressions.json').write_text(json.dumps(regressions, indent=2) + '\n', encoding='utf-8')
        if regressions:
            raise SystemExit(f'Performance regressions above {args.max_regression_pct}%: {regressions}')
    print(f'Benchmark completati: {args.output_dir}')


if __name__ == '__main__':
    main()