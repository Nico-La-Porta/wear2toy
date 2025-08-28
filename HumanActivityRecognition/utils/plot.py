import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

def reliability_plot(confs, preds, labels, num_bins=15, save_path=None):
    """
    Draw a reliability plot (calibration curve) from model predictions and confidences.
    confs: array-like, shape (n_samples,) - predicted confidences (probabilities of predicted class)
    preds: array-like, shape (n_samples,) - predicted class indices
    labels: array-like, shape (n_samples,) - true class indices
    num_bins: int - number of bins
    """
    confs = np.array(confs)
    preds = np.array(preds)
    labels = np.array(labels)
    correct = (preds == labels).astype(int)

    prob_true, prob_pred = calibration_curve(correct, confs, n_bins=num_bins, strategy='uniform')

    plt.figure(figsize=(6, 6))
    plt.plot(prob_pred, prob_true, marker='o', label='Model')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated')
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.title('Reliability Diagram (calibration curve)')
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(f"{save_path}/reliability_plot.png", dpi=400)
    
    plt.close()