"""
Confusion matrix metrics for VAD evaluation.
Provides standardized FPR/FNR calculations and related metrics.
"""
from dataclasses import dataclass
import numpy as np

@dataclass
class Confusion:
    tp: int
    fp: int
    tn: int
    fn: int
    
    @property
    def tpr(self):
        """True Positive Rate (Recall/Sensitivity)"""
        d = (self.tp + self.fn)
        return self.tp / d if d else 0.0

    @property
    def tnr(self):
        """True Negative Rate (Specificity)"""
        d = (self.tn + self.fp)
        return self.tn / d if d else 0.0

    @property
    def fpr(self):
        """False Positive Rate"""
        d = (self.fp + self.tn)
        return self.fp / d if d else 0.0

    @property
    def fnr(self):
        """False Negative Rate"""
        d = (self.fn + self.tp)
        return self.fn / d if d else 0.0

    @property
    def precision(self):
        """Precision"""
        d = (self.tp + self.fp)
        return self.tp / d if d else 0.0

    @property
    def recall(self):
        """Recall (same as TPR)"""
        return self.tpr

    @property
    def f1(self):
        """F1 Score"""
        p, r = self.precision, self.recall
        d = (p + r)
        return (2 * p * r / d) if d else 0.0

    @property
    def accuracy(self):
        """Accuracy"""
        total = self.tp + self.fp + self.tn + self.fn
        return (self.tp + self.tn) / total if total else 0.0


def from_binary(y_true, y_pred):
    """
    Calculate confusion matrix from binary predictions.
    
    Args:
        y_true: Ground truth labels (0/1)
        y_pred: Predicted labels (0/1)
    
    Returns:
        Confusion: Confusion matrix with calculated metrics
    """
    y_true = np.asarray(y_true).astype(np.uint8)
    y_pred = np.asarray(y_pred).astype(np.uint8)
    
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    
    return Confusion(tp, fp, tn, fn)


def from_scores(y_true, y_scores, threshold):
    """
    Calculate confusion matrix from probabilistic scores and threshold.
    
    Args:
        y_true: Ground truth labels (0/1)
        y_scores: Prediction scores/probabilities
        threshold: Decision threshold
    
    Returns:
        Confusion: Confusion matrix with calculated metrics
    """
    y_scores = np.asarray(y_scores)
    y_pred = (y_scores >= threshold).astype(np.uint8)
    return from_binary(y_true, y_pred)
