import time
import threading
import psutil
import pandas as pd
import matplotlib.pyplot as plt

from data_loader import load_ais_data
from parallel_runner import run_parallel_detection
from run_sequential_detection import run_sequential_detection
#from run_parallel_batched import run_parallel_batched <- for testing purposes(putting vessels into batches before sending them to workers)

cpu_usage_log = []
mem_usage_log = []
tracking = False

def track_usage(interval=1):
    global tracking
    process = psutil.Process()
    while tracking:
        cpu = psutil.cpu_percent(interval=None)
        mem = process.memory_info().rss / (1024 * 1024)
        cpu_usage_log.append(cpu)
        mem_usage_log.append(mem)
        time.sleep(interval)

def run_with_tracking(func, df, label=""):
    global tracking, cpu_usage_log, mem_usage_log
    cpu_usage_log = []
    mem_usage_log = []
    tracking = True

    tracker = threading.Thread(target=track_usage)
    tracker.start()

    start = time.time()
    result = func(df)
    elapsed = time.time() - start

    tracking = False
    tracker.join()

    print(f"\n[{label}] Runtime: {elapsed:.2f} seconds")
    if result is not None:
        print(f"[{label}] Total anomalies found: {len(result)}")
        print(f"[{label}] Vessels with anomalies: {result['MMSI'].nunique()}")

    return elapsed, result, cpu_usage_log, mem_usage_log

def plot_usage(cpu_seq, mem_seq, cpu_par, mem_par):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(cpu_seq, label="Sequential", color='gray')
    plt.plot(cpu_par, label="Parallel", color='green')
    plt.title("CPU Usage Over Time")
    plt.ylabel("CPU %")
    plt.xlabel("Seconds")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(mem_seq, label="Sequential", color='gray')
    plt.plot(mem_par, label="Parallel", color='green')
    plt.title("Memory Usage Over Time")
    plt.ylabel("Memory (MB)")
    plt.xlabel("Seconds")
    plt.legend()

    plt.tight_layout()
    plt.savefig("benchmark_cpu_memory_usage.png")
    print("Saved resource usage plot to 'benchmark_cpu_memory_usage.png'")

def main():
    DATA_PATH = "C:/Users/lukan/Downloads/aisdk-2025-03-14/aisdk-2025-03-14.csv"
    df = load_ais_data(DATA_PATH)

    print("\nRunning SEQUENTIAL detection...")
    t_seq, result_seq, cpu_seq, mem_seq = run_with_tracking(run_sequential_detection, df, label="Sequential")

#     print("\nRunning PARALLEL (batched) detection...")
#     t_par, result_par, cpu_par, mem_par = run_with_tracking(
#         lambda df: run_parallel_batched(df, batch_size=151),
#         df, label="Parallel (batched)"
# ) #Only used for testing batching..
    print("\nRunning PARALLEL detection...")
    t_par, result_par, cpu_par, mem_par = run_with_tracking(
    run_parallel_detection,
    df,
    label="Parallel"
)

    speedup = t_seq / t_par if t_par > 0 else float("inf")
    print(f"\n Speedup: {speedup:.2f}x")

    plt.figure(figsize=(6, 4))
    plt.bar(["Sequential", "Parallel"], [t_seq, t_par], color=['gray', 'green'])
    plt.title("Execution Time")
    plt.ylabel("Time (s)")
    plt.savefig("benchmark_execution_time.png")
    print("Saved runtime chart to 'benchmark_execution_time.png'")

    plot_usage(cpu_seq, mem_seq, cpu_par, mem_par)

if __name__ == "__main__":
    main()
