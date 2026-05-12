from Kmeans import KMeans
from KNN import KNN

import time
import matplotlib.pyplot as plt
import numpy as np


def knn_accuracy(
    knn: KNN, test_data: np.ndarray, test_labels: np.ndarray, k: int
) -> float:
    predictions = knn.predict(test_data, k)
    correct = np.sum(predictions == test_labels)
    return correct / len(test_labels)


def precision_per_class(
    predictions: np.ndarray, true_labels: np.ndarray, classes: list
) -> dict:
    precision = {}
    for pclass in classes:
        tp = np.sum((predictions == pclass) & (true_labels == pclass))
        fp = np.sum((predictions == pclass) & (true_labels != pclass))
        if tp + fp == 0:
            precision[pclass] = 0.0
        else:
            precision[pclass] = tp / (tp + fp)
    return precision


def recall_per_class(
    predictions: np.ndarray, true_labels: np.ndarray, classes: list
) -> dict:
    recall = {}
    for cls in classes:
        tp = np.sum((predictions == cls) & (true_labels == cls))
        fn = np.sum((predictions != cls) & (true_labels == cls))
        if tp + fn == 0:
            recall[cls] = 0.0
        else:
            recall[cls] = tp / (tp + fn)
    return recall


def f1_score_per_class(
    predictions: np.ndarray, true_labels: np.ndarray, classes: list
) -> dict:
    prec = precision_per_class(predictions, true_labels, classes)
    rec = recall_per_class(predictions, true_labels, classes)

    f1 = {}
    for cls in classes:
        p, r = prec[cls], rec[cls]
        if p + r == 0:
            f1[cls] = 0.0
        else:
            f1[cls] = 2 * p * r / (p + r)
    return f1


def retrieval_precision(
    knn: KNN, test_data: np.ndarray, test_labels: np.ndarray, k: int
) -> float:
    knn.get_k_neighbours(test_data, k)
    precisions = []
    for i, neighbours in enumerate(knn.neighbors):
        relevant = np.sum(neighbours == test_labels[i])
        precisions.append(relevant / k)
    return float(np.mean(precisions))


def retrieval_precision_per_class(
    knn: KNN, test_data: np.ndarray, test_labels: np.ndarray, k: int, classes: list
) -> dict:
    knn.get_k_neighbours(test_data, k)
    precision_dict = {}
    for cls in classes:
        idxs = np.where(test_labels == cls)[0]
        if len(idxs) == 0:
            precision_dict[cls] = 0.0
            continue
        precisions = []
        for i in idxs:
            relevant = np.sum(knn.neighbors[i] == cls)
            precisions.append(relevant / k)
        precision_dict[cls] = float(np.mean(precisions))
    return precision_dict


def kmean_statistics(X: np.ndarray, Kmax: int, options: dict = {}):
    ks, wcds, iters, times = [], [], [], []

    for K in range(2, Kmax + 1):
        km = KMeans(X, K, options)
        t0 = time.time()
        km.fit()
        elapsed = time.time() - t0
        km.withinClassDistance()

        ks.append(K)
        wcds.append(km.WCD)
        iters.append(km.num_iter)
        times.append(elapsed)

        print(f"  K={K:2d}  WCD={km.WCD:.4f}  iters={km.num_iter:3d}  t={elapsed:.3f}s")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("KMeans Statistics", fontsize=13, fontweight="bold")

    axes[0].plot(ks, wcds, "o-", color="steelblue")
    axes[0].set_title("Within Class Distance (WCD)")
    axes[0].set_xlabel("K")
    axes[0].set_ylabel("WCD")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(ks, iters, "s-", color="darkorange")
    axes[1].set_title("Iteracions fins convergència")
    axes[1].set_xlabel("K")
    axes[1].set_ylabel("Iteracions")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(ks, times, "^-", color="seagreen")
    axes[2].set_title("Temps d'execució")
    axes[2].set_xlabel("K")
    axes[2].set_ylabel("Temps (s)")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return {"K": ks, "WCD": wcds, "iters": iters, "times": times}


def get_shape_accuracy(
    predicted_labels: np.ndarray, true_labels: np.ndarray, classes: list = []
) -> dict:
    if classes is None or len(classes) == 0:
        classes = sorted(set(list(predicted_labels) + list(true_labels)))

    acc = float(np.mean(predicted_labels == true_labels))

    precision, recall, f1 = {}, {}, {}
    for cls in classes:
        tp = np.sum((predicted_labels == cls) & (true_labels == cls))
        fp = np.sum((predicted_labels == cls) & (true_labels != cls))
        fn = np.sum((predicted_labels != cls) & (true_labels == cls))

        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

        precision[cls] = p
        recall[cls] = r
        f1[cls] = f

    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


def get_color_accuracy(predicted_colors: list, true_colors: list) -> dict:
    jaccards, f1s, exact = [], [], []

    for pred, gt in zip(predicted_colors, true_colors):
        pred_set = set([c.lower() for c in pred])
        gt_set = set([c.lower() for c in gt])

        inter = len(pred_set & gt_set)
        union = len(pred_set | gt_set)

        jacc = inter / union if union > 0 else 0.0
        f1 = (
            2 * inter / (len(pred_set) + len(gt_set))
            if (len(pred_set) + len(gt_set)) > 0
            else 0.0
        )
        ex = 1.0 if pred_set == gt_set else 0.0

        jaccards.append(jacc)
        f1s.append(f1)
        exact.append(ex)

    return {
        "mean_jaccard": float(np.mean(jaccards)),
        "mean_f1": float(np.mean(f1s)),
        "exact_match": float(np.mean(exact)),
        "per_image": [
            {"jaccard": j, "f1": f, "exact": e} for j, f, e in zip(jaccards, f1s, exact)
        ],
    }


def make_report(
    knn: KNN, test_data: np.ndarray, test_labels: np.ndarray, classes: list, k: int
):
    predictions = knn.predict(test_data, k)

    acc = knn_accuracy(knn, test_data, test_labels, k)
    prec = precision_per_class(predictions, test_labels, classes)
    rec = recall_per_class(predictions, test_labels, classes)
    f1 = f1_score_per_class(predictions, test_labels, classes)
    retrieval_global = retrieval_precision(knn, test_data, test_labels, k)
    retrieval_cls = retrieval_precision_per_class(
        knn, test_data, test_labels, k, classes
    )

    print("=" * 60)
    print(f"  INFORME QUANTITATIU — KNN (k={k})")
    print("=" * 60)
    print(f"\n  Accuracy global: {acc:.4f} ({acc*100:.2f}%)\n")

    print(
        f"  {'Classe':<20} {'Precisió':>10} {'Recall':>10} {'F1-score':>10} {'Retrieval@K':>12}"
    )
    print("  " + "-" * 64)
    for cls in sorted(classes):
        print(
            f"  {cls:<20} {prec[cls]:>10.4f} {rec[cls]:>10.4f} {f1[cls]:>10.4f} {retrieval_cls[cls]:>12.4f}"
        )

    print("  " + "-" * 64)
    macro_prec = np.mean(list(prec.values()))
    macro_rec = np.mean(list(rec.values()))
    macro_f1 = np.mean(list(f1.values()))
    print(
        f"  {'MACRO AVG':<20} {macro_prec:>10.4f} {macro_rec:>10.4f} {macro_f1:>10.4f} {retrieval_global:>12.4f}"
    )
    print("=" * 60)
