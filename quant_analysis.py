import numpy as np
from time import time
import matplotlib.pyplot as plt

from Kmeans import KMeans

def kmeans_statistics(images: np.ndarray, Kmax: int) -> dict:
    results = {
        "k": [],
        "wcd": [],
        "time": [],
        "iters": [],
    }

    for k in range(2, Kmax + 1):
        kmeans_obj.K = k

        start_time = time()
        kmeans_obj.fit()
        end_time = time()

        results["k"].append(k)
        results["wcd"].append(kmeans_obj.withinClassDistance())
        results["iters"].append(kmeans_obj.num_iter)
        results["time"].append(end_time - start_time)

    plot_stats(results)
    return results

def plot_stats(results):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.plot(results["k"], results["wcd"], marker="o")
    plt.title("WCD vs K (Elbow methdod)")
    plt.xlabel("K")

    plt.subplot(1, 3, 2)
    plt.plot(results["k"], results["time"], marker="^", color="orange")
    plt.title("Time vs K")

    plt.subplot(1, 3, 3)
    plt.plot(results["k"], results["iters"], marker="*", color="green")
    plt.title("Iterations vs K")

    plt.tight_layout()
    plt.show()
