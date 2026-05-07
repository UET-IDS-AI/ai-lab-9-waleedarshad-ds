"""
AI_stats_lab.py

Lab: Training and Evaluating Classification Models

Topics:
- Confusion matrix
- Recall
- Fallout
- Precision
- Accuracy
- Thresholding prediction scores
- Effect of changing threshold
- Training two classifiers
- Comparing model performance

Instructions:
- Implement all functions.
- Do NOT change function names.
- Do NOT print inside functions.
- Return exactly the required formats.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


# Question 1: Confusion Matrix, Metrics, and Threshold Effects

def confusion_matrix_counts(y_true, y_pred):
    """
    Compute confusion matrix counts for binary classification.
    Returns: (TP, FP, FN, TN)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    TP = np.sum((y_true == 1) & (y_pred == 1))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    TN = np.sum((y_true == 0) & (y_pred == 0))

    return (int(TP), int(FP), int(FN), int(TN))


def classification_metrics(y_true, y_pred):
    """
    Compute classification metrics.
    Returns a dictionary with recall, fallout, precision, and accuracy.
    """
    TP, FP, FN, TN = confusion_matrix_counts(y_true, y_pred)

    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    fallout = FP / (FP + TN) if (FP + TN) > 0 else 0.0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    accuracy = (TP + TN) / (TP + FP + FN + TN) if (TP + FP + FN + TN) > 0 else 0.0

    return {
        "recall": float(recall),
        "fallout": float(fallout),
        "precision": float(precision),
        "accuracy": float(accuracy)
    }


def apply_threshold(scores, threshold):
    """
    Convert prediction scores into binary predictions.
    1 if score >= threshold, 0 otherwise.
    """
    scores = np.array(scores)
    return (scores >= threshold).astype(int)


def threshold_metrics_analysis(y_true, scores, thresholds):
    """
    Analyze how changing threshold affects recall and fallout.
    """
    analysis_results = []
    
    for t in thresholds:
        y_pred = apply_threshold(scores, t)
        metrics = classification_metrics(y_true, y_pred)
        
        # Build the dictionary for this threshold
        result = {"threshold": t}
        result.update(metrics)
        analysis_results.append(result)
        
    return analysis_results


# Question 2: Train Two Classifiers and Evaluate Them


def train_two_classifiers(X_train, y_train):
    """
    Train Logistic Regression and Decision Tree.
    """
    # Initialize models with required parameters
    log_reg = LogisticRegression(max_iter=1000)
    dt_clf = DecisionTreeClassifier(random_state=0)

    log_reg.fit(X_train, y_train)
    dt_clf.fit(X_train, y_train)

    return {
        "logistic_regression": log_reg,
        "decision_tree": dt_clf
    }


def evaluate_classifier(model, X_test, y_test, threshold=0.5):
    """
    Evaluate a trained classifier using a specific threshold.
    """
    probs = model.predict_proba(X_test)[:, 1]

    y_pred = apply_threshold(probs, threshold)

    TP, FP, FN, TN = confusion_matrix_counts(y_test, y_pred)

    metrics = classification_metrics(y_test, y_pred)
 
    results = {
        "TP": TP, "FP": FP, "FN": FN, "TN": TN
    }
    results.update(metrics)
    
    return results


def compare_classifiers(X_train, y_train, X_test, y_test, threshold=0.5):
    """
    Train two classifiers and evaluate both on the same test set.
    """
    models = train_two_classifiers(X_train, y_train)
    
    lr_eval = evaluate_classifier(models["logistic_regression"], X_test, y_test, threshold)
    dt_eval = evaluate_classifier(models["decision_tree"], X_test, y_test, threshold)
    
    return {
        "logistic_regression": lr_eval,
        "decision_tree": dt_eval
    }


if __name__ == "__main__":
    print("Implement all required functions.")
