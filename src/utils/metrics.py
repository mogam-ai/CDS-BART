import torch
from evaluate import load


def compute_regression_metrics(pred):
    labels = pred.label_ids
    predictions = (
        pred.predictions[0] if len(pred.predictions) == 2 else pred.predictions
    )

    metric1 = load("r_squared", trust_remote_code=True)
    metric2 = load("mae", trust_remote_code=True)
    metric3 = load("mse", trust_remote_code=True)
    metric4 = load("spearmanr", trust_remote_code=True)

    r2 = metric1.compute(predictions=predictions, references=labels)
    mae = metric2.compute(predictions=predictions, references=labels)["mae"]
    mse = metric3.compute(predictions=predictions, references=labels)["mse"]
    spearmanr = metric4.compute(predictions=predictions, references=labels)["spearmanr"]

    return {
        "r2": r2,
        "mae": mae,
        "mse": mse,
        "spearmanr": spearmanr,
    }


def compute_multi_class_classification_metrics(pred):
    """
    Compute various metrics for multi-class classification.

    Args:
    pred: The predictions object containing predicted labels and scores.

    Returns:
    dict: A dictionary containing the computed metrics:
    - auroc: Area Under the Receiver Operating Characteristic curve.
    - precision: Precision of the predictions.
    - recall: Recall of the predictions.
    - f1: F1 score of the predictions.
    - accuracy: Accuracy of the predictions.
    """
    labels = pred.label_ids

    predictions = torch.Tensor(
        pred.predictions[0] if len(pred.predictions) == 2 else pred.predictions
    )
    probability = torch.softmax(predictions, dim=-1).numpy()[:, 1]
    predictions = torch.argmax(predictions, axis=-1).numpy()

    # auroc_metric = load("roc_auc", trust_remote_code=True)
    # auprc_metric = average_precision_score
    precision_metric = load("precision", trust_remote_code=True)
    recall_metric = load("recall", trust_remote_code=True)
    f1_metric = load("f1", trust_remote_code=True)
    accuracy_metric = load("accuracy", trust_remote_code=True)

    # auroc = auroc_metric.compute(prediction_scores=probability, references=labels)["roc_auc"]
    # auprc = auprc_metric(labels, probability)
    precision = precision_metric.compute(
        predictions=predictions, references=labels, average="macro"
    )["precision"]
    recall = recall_metric.compute(
        predictions=predictions, references=labels, average="macro"
    )["recall"]
    f1 = f1_metric.compute(predictions=predictions, references=labels, average="macro")[
        "f1"
    ]
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)[
        "accuracy"
    ]

    return {
        # "auroc": auroc,
        # "auprc": auprc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }
