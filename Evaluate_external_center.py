from sklearn.metrics import (roc_curve, auc, confusion_matrix,accuracy_score,
                             precision_score, recall_score, roc_auc_score)
from Bootstrap import bootstrap
from libtlda.iw import ImportanceWeightedClassifier
from libtlda.suba import SubspaceAlignedClassifier
import numpy as np
import os


def evaluate_on_other_center(model, X_test_ext, y_test_ext, train_center, test_center, threshold, sampler,center_output_dir, fname="a.txt"):
    model_name = type(model).__name__

    test_proba = model.predict_proba(X_test_ext)[:, 1]
    test_pred = (test_proba >= threshold).astype(int)
    fpr, tpr, thresholds = roc_curve(y_test_ext, test_proba)
    cm = confusion_matrix(y_test_ext, test_pred)

    auc_score = roc_auc_score(y_test_ext, test_proba)
    sensitivity = recall_score(y_test_ext, test_pred)
    specificity = recall_score(y_test_ext, test_pred, pos_label=0)
    precision = precision_score(y_test_ext, test_pred, zero_division=0)
    test_acc = accuracy_score(y_test_ext, test_pred)

    metrics_ci, roc_ci = bootstrap(model, X_test_ext, y_test_ext, threshold, n_bootstrap=1000)
    fi = open(fname, 'a')
    print(f'Best AUC of PointEstimate in Bootstrap on external center {test_center} with {sampler} augmentation: {auc_score}',file=fi)
    print(f'Best AUC of PointEstimate in Bootstrap on external center {test_center} with {sampler} augmentation: {metrics_ci["auc"]},',file=fi)
    fi.close()
    figdata = [model_name, [test_proba, test_pred, y_test_ext, auc_score, fpr, tpr, cm]]
    np.save(os.path.join(center_output_dir, f"Int_{train_center}_Ext_{test_center}_{model_name}_{sampler}"), np.array(figdata, dtype=object))
    return {
        'train_center': train_center,
        'test_center': test_center,
        'auc': metrics_ci['auc'],
        'sensitivity': metrics_ci['sensitivity'],
        'specificity': metrics_ci['specificity'],
        'precision': metrics_ci['precision'],
        'accuracy': metrics_ci['accuracy'],
        'threshold': threshold,
        'roc_ci': roc_ci
    }

def evaluate_on_other_center_adapt_iw(n_est, max_dep, threshold, X_train, y_train, X_train_ext, X_test, y_test, center_name, ext_center, sampler,center_output_dir, fname="a.txt"):
    model_name = 'ImportanceWeighted'
    transfer_model = ImportanceWeightedClassifier('gini', n_estimators= n_est.astype(int), max_depth = max_dep.astype(int))
    transfer_model.fit(X_train, y_train, X_train_ext)
    # Testing
    test_proba = transfer_model.predict_proba(X_test)[:, 1]
    test_pred = (test_proba >= threshold).astype(int)

    fpr, tpr, thresholds = roc_curve(y_test, test_proba)
    cm = confusion_matrix(y_test, test_pred)

    auc_score = roc_auc_score(y_test, test_proba)
    sensitivity = recall_score(y_test, test_pred)
    specificity = recall_score(y_test, test_pred, pos_label=0)
    precision = precision_score(y_test, test_pred, zero_division=0)
    test_acc = accuracy_score(y_test, test_pred)

    metrics_ci, roc_ci = bootstrap(transfer_model, X_test, y_test, threshold, n_bootstrap=1000)
    fi = open(fname, 'a')
    print(f'Best AUC of PointEstimate in Bootstrap on external center {ext_center} with {sampler} augmentation: {auc_score}',file=fi)
    print(f'Best AUC of PointEstimate in Bootstrap on external center {ext_center} with {sampler} augmentation: {metrics_ci["auc"]},',file=fi)
    fi.close()
    figdata = [model_name, [test_proba, test_pred, y_test, auc_score, fpr, tpr, cm]]
    np.save(os.path.join(center_output_dir, f"Int_{center_name}_Ext_{ext_center}_{model_name}_{sampler}"), np.array(figdata, dtype=object))
    return {
        'train_center': center_name,
        'test_center': ext_center,
        'auc': metrics_ci['auc'],
        'sensitivity': metrics_ci['sensitivity'],
        'specificity': metrics_ci['specificity'],
        'precision': metrics_ci['precision'],
        'accuracy': metrics_ci['accuracy'],
        'threshold': threshold,
        'roc_ci': roc_ci
    }


def evaluate_on_other_center_adapt_ss(l2ss, threshold, X_train, y_train, X_train_ext, X_test, y_test, center_name, ext_center, sampler,center_output_dir, fname="a.txt"):
    model_name = 'SubspaceAligned'
    transfer_model = SubspaceAlignedClassifier(l2_regularization=l2ss, subspace_dim= 15)
    transfer_model.fit(X_train, y_train, X_train_ext)

    # Testing
    test_proba = transfer_model.predict_proba(X_test)[:, 1]
    test_pred = (test_proba >= threshold).astype(int)
    fpr, tpr, thresholds = roc_curve(y_test, test_proba)
    cm = confusion_matrix(y_test, test_pred)

    auc_score = roc_auc_score(y_test, test_proba)
    sensitivity = recall_score(y_test, test_pred)
    specificity = recall_score(y_test, test_pred, pos_label=0)
    precision = precision_score(y_test, test_pred, zero_division=0)
    test_acc = accuracy_score(y_test, test_pred)

    metrics_ci, roc_ci = bootstrap(transfer_model, X_test, y_test, threshold, n_bootstrap=1000)
    fi = open(fname, 'a')
    print(f'Best AUC of PointEstimate in Bootstrap on external center {ext_center} with {sampler} augmentation: {auc_score}',file=fi)
    print(f'Best AUC of PointEstimate in Bootstrap on external center {ext_center} with {sampler} augmentation: {metrics_ci["auc"]},',file=fi)
    fi.close()
    figdata = [model_name, [test_proba, test_pred, y_test, auc_score, fpr, tpr, cm]]
    np.save(os.path.join(center_output_dir, f"Int_{center_name}_Ext_{ext_center}_{model_name}_{sampler}"), np.array(figdata, dtype=object))
    return {
        'train_center': center_name,
        'test_center': ext_center,
        'auc': metrics_ci['auc'],
        'sensitivity': metrics_ci['sensitivity'],
        'specificity': metrics_ci['specificity'],
        'precision': metrics_ci['precision'],
        'accuracy': metrics_ci['accuracy'],
        'threshold': threshold,
        'roc_ci': roc_ci
    }
