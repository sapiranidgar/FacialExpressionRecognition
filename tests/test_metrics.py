import numpy as np

from fer.training.metrics import compute_metrics


def test_compute_metrics_per_class():
    y_true = np.array([0, 0, 1, 1, 2, 2, 2])
    y_pred = np.array([0, 1, 1, 1, 2, 2, 0])

    metrics = compute_metrics(y_true, y_pred, num_classes=3)
    per_class = metrics["per_class"]

    assert per_class[0]["precision"] == 0.5
    assert per_class[0]["recall"] == 0.5
    assert per_class[0]["support"] == 2

    assert abs(per_class[1]["precision"] - (2 / 3)) < 1e-9
    assert per_class[1]["recall"] == 1.0
    assert per_class[1]["support"] == 2

    assert per_class[2]["precision"] == 1.0
    assert abs(per_class[2]["recall"] - (2 / 3)) < 1e-9
    assert per_class[2]["support"] == 3

    # one-vs-rest accuracy for class 0: true==0 -> [T,T,F,F,F,F,F], pred==0 -> [T,F,F,F,F,F,T]
    # matches at 5 of 7 positions
    assert abs(per_class[0]["accuracy"] - (5 / 7)) < 1e-9
