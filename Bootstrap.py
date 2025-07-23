from numpy import interp
import numpy as np
import pandas as pd
from sklearn.metrics import (roc_curve, auc, confusion_matrix, accuracy_score,
                             precision_score, recall_score, roc_auc_score)


def bootstrap(clf, X, y, threshold, n_bootstrap=1000):
    if isinstance(X, pd.DataFrame):
        feature_names = X.columns.tolist()
        X_values = X.values
    else:
        feature_names = None
        X_values = X

    if isinstance(y, pd.Series):
        y_values = y.values
    else:
        y_values = y

    metrics = {
        'auc': [],
        'sensitivity': [],
        'specificity': [],
        'precision': [],
        'accuracy':[]
    }
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    n_samples = len(X)
    valid_iterations = 0
    for b in range(n_bootstrap):
        idx = np.random.choice(np.arange(n_samples), size=n_samples, replace=True).astype(int)
        X_boot = X_values[idx]
        y_boot = y_values[idx]
        unique_classes = np.unique(y_boot)
        if len(unique_classes) < 2:
            continue
        if feature_names is not None:
            X_boot_df = pd.DataFrame(X_boot, columns=feature_names)
            prob = clf.predict_proba(X_boot_df)[:, 1]
        else:
            prob = clf.predict_proba(X_boot)[:, 1]
        if np.any(np.isnan(prob)) or np.any(np.isinf(prob)):
            continue
        pred = (prob >= threshold).astype(int)
        roc_auc = roc_auc_score(y_boot, prob)
        if np.isnan(roc_auc):
            continue
        sensitivity = recall_score(y_boot, pred)
        specificity = recall_score(y_boot, pred, pos_label=0)
        precision = precision_score(y_boot, pred, zero_division=0)
        accuracy = accuracy_score(y_boot, pred)

        metrics['auc'].append(roc_auc)
        metrics['sensitivity'].append(sensitivity)
        metrics['specificity'].append(specificity)
        metrics['precision'].append(precision)
        metrics['accuracy'].append(accuracy)

        fpr, tpr, _ = roc_curve(y_boot, prob)
        interp_tpr = interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(auc(fpr, tpr))
        valid_iterations += 1

    if feature_names is not None:
        prob_full = clf.predict_proba(X)[:, 1]
    else:
        prob_full = clf.predict_proba(X_values)[:, 1]

    pred_full = (prob_full >= threshold).astype(int)

    point_estimates = {
        'auc': roc_auc_score(y_values, prob_full),
        'sensitivity': recall_score(y_values, pred_full),
        'specificity': recall_score(y_values, pred_full, pos_label=0),
        'precision': precision_score(y_values, pred_full, zero_division=0),
        'accuracy': accuracy_score(y_values, pred_full)
    }
    results = {}
    roc_ci = {}
    for metric, values in metrics.items():
        if len(values) > 0:
            ci_lower = np.percentile(values, 2.5)
            ci_upper = np.percentile(values, 97.5)
            median = np.percentile(values, 50)
            results[metric] = {
                'point_estimate': point_estimates[metric],
                'median': median,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'n_valid': len(values)
            }
    if len(tprs) > 0:
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + 1.96 * std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - 1.96 * std_tpr, 0)
        roc_ci = {
            'mean_fpr': mean_fpr,
            'mean_tpr': mean_tpr,
            'tprs_lower': tprs_lower,
            'tprs_upper': tprs_upper,
            'mean_auc': mean_auc,
            'auc_ci_lower': np.percentile(aucs, 2.5) if len(aucs) > 0 else np.nan,
            'auc_ci_upper': np.percentile(aucs, 97.5) if len(aucs) > 0 else np.nan,
            'n_valid': len(tprs)
        }
    return results, roc_ci
