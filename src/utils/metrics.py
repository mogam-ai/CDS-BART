import numpy as np
from scipy.special import softmax
from scipy.stats import spearmanr
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize


def compute_regression_metrics(eval_pred):
    if isinstance(eval_pred.predictions, tuple):
        predictions = eval_pred.predictions[0]
    else:
        predictions = eval_pred.predictions

    labels = eval_pred.label_ids

    predictions = predictions.reshape(-1)
    labels = labels.reshape(-1)

    mse = mean_squared_error(labels, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(labels, predictions)
    r2 = r2_score(labels, predictions)
    spearman_corr, _ = spearmanr(predictions, labels)
    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2, "spearmanr": spearman_corr}


def compute_classification_metrics(eval_pred):
    """
    Computes classification metrics for either binary or multi-class tasks.

    This function is designed to be used with the Hugging Face Trainer. It
    automatically detects the number of classes from the model's output logits
    and adjusts the metric calculations accordingly.

    Args:
        eval_pred: An `EvalPrediction` object from the Hugging Face Trainer.
                   It's a tuple containing logits and label_ids.

    Returns:
        dict: A dictionary of computed metrics (e.g., accuracy, f1, auroc).
    """

    logits, labels = eval_pred.predictions, eval_pred.label_ids

    if isinstance(logits, tuple):
        logits = logits[0]

    predictions = np.argmax(logits, axis=-1)
    probabilities = softmax(logits, axis=1)

    num_labels = logits.shape[1]

    # Initialize the metrics dictionary
    metrics = {}

    # --- METRICS CALCULATION ---

    if num_labels == 2:
        # --- BINARY CLASSIFICATION ---
        positive_class_probs = probabilities[:, 1]

        # Use 'binary' averaging for binary-specific metrics
        metrics["accuracy"] = accuracy_score(labels, predictions)
        metrics["f1"] = f1_score(labels, predictions, average="binary")
        metrics["recall"] = recall_score(labels, predictions, average="binary")
        metrics["precision"] = precision_score(labels, predictions, average="binary")

        # AUROC and AUPRC for binary classification
        metrics["auroc"] = roc_auc_score(labels, positive_class_probs)
        metrics["auprc"] = average_precision_score(labels, positive_class_probs)

    else:
        # --- MULTI-CLASS CLASSIFICATION ---
        avg_method = "macro"

        metrics["accuracy"] = accuracy_score(labels, predictions)
        metrics["f1_macro"] = f1_score(labels, predictions, average=avg_method)
        metrics["recall_macro"] = recall_score(labels, predictions, average=avg_method)
        metrics["precision_macro"] = precision_score(
            labels, predictions, average=avg_method
        )

        labels_binarized = label_binarize(labels, classes=np.arange(num_labels))

        metrics["auroc_macro"] = roc_auc_score(
            labels, probabilities, multi_class="ovr", average="macro"
        )
        metrics["auroc_weighted"] = roc_auc_score(
            labels, probabilities, multi_class="ovr", average="weighted"
        )

        metrics["auprc_macro"] = average_precision_score(
            labels_binarized, probabilities, average="macro"
        )
        metrics["auprc_weighted"] = average_precision_score(
            labels_binarized, probabilities, average="weighted"
        )

    return metrics
