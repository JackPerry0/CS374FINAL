import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


def plot_results(y_true, y_pred, title="Model Results", logscale=False):

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    residuals = y_pred - y_true

    corr = pearsonr(y_pred, y_true)[0]
    mae = np.mean(np.abs(residuals))
    medae = np.median(np.abs(residuals))
    std_err = np.std(residuals)

    plt.figure(figsize=(16, 10))
    plt.suptitle(
        f"{title}\nCorr={corr:.4f}, MAE={mae:.3f}, MedAE={medae:.3f}, Std={std_err:.3f}",
        fontsize=14
    )

    # 1. True vs Predicted
    plt.subplot(2, 2, 1)
    plt.scatter(y_true, y_pred, s=6, alpha=0.6)
    plt.xlabel("True Flux")
    plt.ylabel("Predicted Flux")
    plt.title("True vs Predicted")
    if logscale:
        plt.xscale("log")
        plt.yscale("log")

    # 2. Residual Histogram
    plt.subplot(2, 2, 2)
    plt.hist(residuals, bins=100, color="gray", edgecolor="black")
    plt.title("Residual Distribution")
    plt.xlabel("Residual = Pred - True")
    plt.ylabel("Count")

    # 3. Residuals vs Predicted (heteroscedasticity check)
    plt.subplot(2, 2, 3)
    plt.scatter(y_pred, residuals, s=6, alpha=0.6)
    plt.axhline(0, color="red", linestyle="--")
    plt.xlabel("Predicted Flux")
    plt.ylabel("Residual")
    plt.title("Residuals vs Predicted")
    if logscale:
        plt.xscale("log")

    # 4. Distribution comparison
    plt.subplot(2, 2, 4)
    plt.hist(y_true, bins=100, color="blue", alpha=0.5, label="True")
    plt.hist(y_pred, bins=100, color="orange", alpha=0.5, label="Predicted")
    plt.legend()
    plt.title("Flux Distribution Comparison")
    plt.xlabel("Flux")
    if logscale:
        plt.xscale("log")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()



def plot_multiple_models(y_true, preds_dict, title_prefix="Comparison", logscale=False):
    """
    Compare multiple models on the same y_true.
    
    preds_dict: {
        "FULL Ridge": y_pred_array,
        "FULL NN": y_pred_array,
        "ENR Ridge": y_pred_array,
        ...
    }
    """

    plt.figure(figsize=(12, 6))
    plt.title(title_prefix + " - Flux Distributions")

    for name, pred in preds_dict.items():
        plt.hist(pred, bins=100, alpha=0.5, label=name)

    plt.hist(y_true, bins=100, color="black", alpha=0.5, label="True")
    plt.legend()
    plt.xlabel("Flux")
    if logscale:
        plt.xscale("log")
    plt.show()
