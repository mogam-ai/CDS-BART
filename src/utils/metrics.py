import numpy as np
import torch as th
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
    # 튜플에서 예측값 추출
    if isinstance(eval_pred.predictions, tuple):
        predictions = eval_pred.predictions[0]
    else:
        predictions = eval_pred.predictions

    probability = th.softmax(predictions, dim=-1).numpy()[:, 1]
    predictions = th.argmax(predictions, axis=-1).numpy()
    labels = eval_pred.label_ids

    predictions = predictions.reshape(-1)
    labels = labels.reshape(-1)

    auroc = roc_auc_score(labels, probability, multi_class="ovr")
    auprc = average_precision_score(labels, probability)
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="macro")
    accuracy = accuracy_score(labels, predictions)
    return {
        "auroc": auroc,
        "auprc": auprc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }
