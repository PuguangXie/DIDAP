import os
import numpy as np
from sklearn import clone
from sklearn.metrics import (roc_curve, auc, confusion_matrix, accuracy_score,
                             precision_score, recall_score, roc_auc_score)
from libtlda.iw import ImportanceWeightedClassifier
from libtlda.suba import SubspaceAlignedClassifier
from Augmentation import augmentation
from Bootstrap import bootstrap
from Save_result import save_results, save_best_model


def single_center_cv(center_output_dir, model, X_train, y_train, X_test, y_test, center_name, sampler, random_state=1,fname="a.txt"):

    model_name = type(model).__name__

    # dis_flag: Whether to do sample distribution analysis
    # Augmentation
    X_aug, y_aug = augmentation(center_name, X_train, y_train, sampler, dis_flag=True, random_state=random_state)

    # Training
    final_model = clone(model)
    final_model.fit(X_aug, y_aug)

    train_proba = final_model.predict_proba(X_train)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_train, train_proba)
    youden_scores = tpr - fpr
    max_idx = np.argmax(youden_scores)
    best_threshold = thresholds[max_idx] if max_idx < len(thresholds) else 0.5

    # Testing
    test_proba = final_model.predict_proba(X_test)[:, 1]
    test_pred = (test_proba >= best_threshold).astype(int)
    fpr, tpr, thresholds = roc_curve(y_test, test_proba)
    cm = confusion_matrix(y_test, test_pred)

    test_auc = roc_auc_score(y_test, test_proba)
    test_sensitivity = recall_score(y_test, test_pred)
    test_specificity = recall_score(y_test, test_pred, pos_label=0)
    test_precision = precision_score(y_test, test_pred, zero_division=0)
    test_acc = accuracy_score(y_test, test_pred)

    # Bootstrap
    metrics_ci, roc_ci = bootstrap(final_model, X_test, y_test, best_threshold, n_bootstrap=1000)
    test_results = {
        'auc': metrics_ci['auc'],
        'sensitivity': metrics_ci['sensitivity'],
        'specificity': metrics_ci['specificity'],
        'precision': metrics_ci['precision'],
        'accuracy': metrics_ci['accuracy'],
        'threshold': best_threshold,
        'roc_ci': roc_ci
    }
    fi = open(fname, 'a')
    print(f'AUC of the overall Testing on {center_name} with {sampler} augmentation: {test_auc}',file=fi)
    print(f'Best AUC of PointEstimate in Bootstrap on {center_name} with {sampler} augmentation: {metrics_ci["auc"]},',file=fi)
    fi.close()
    figdata = [model_name, [test_proba, test_pred, y_test, test_auc, fpr, tpr, cm]]
    np.save(os.path.join(center_output_dir, f"{center_name}_{model_name}_{sampler}"), np.array(figdata, dtype=object))
    save_results(test_results, y_test, test_proba, test_pred, model_name, center_name, sampler, center_output_dir)
    return test_results, final_model, best_threshold

def single_center_cv_adapt_iw(center_output_dir, n_est, max_dep, X_train, y_train, X_test, y_test, center_name, sampler, random_state=1, fname="a.txt"):
    model_name = 'ImportanceWeighted'
    # Augmentation
    X_aug, y_aug = augmentation(center_name, X_train, y_train, sampler, dis_flag=False, random_state=random_state)
    # Training
    model = ImportanceWeightedClassifier('gini', n_estimators= n_est, max_depth = max_dep)
    model.fit(X_aug, y_aug, X_aug)

    train_proba = model.predict_proba(X_train)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_train, train_proba)
    youden_scores = tpr - fpr
    max_idx = np.argmax(youden_scores)
    best_threshold = thresholds[max_idx] if max_idx < len(thresholds) else 0.5

    # Testing
    test_proba = model.predict_proba(X_test)[:, 1]
    test_pred = (test_proba >= best_threshold).astype(int)
    fpr, tpr, thresholds = roc_curve(y_test, test_proba)
    cm = confusion_matrix(y_test, test_pred)

    test_auc = roc_auc_score(y_test, test_proba)
    test_sensitivity = recall_score(y_test, test_pred)
    test_specificity = recall_score(y_test, test_pred, pos_label=0)
    test_precision = precision_score(y_test, test_pred, zero_division=0)
    test_acc = accuracy_score(y_test, test_pred)

    # Bootstrap
    metrics_ci, roc_ci = bootstrap(model, X_test, y_test, best_threshold, n_bootstrap=1000)
    test_results = {
        'auc': metrics_ci['auc'],
        'sensitivity': metrics_ci['sensitivity'],
        'specificity': metrics_ci['specificity'],
        'precision': metrics_ci['precision'],
        'accuracy': metrics_ci['accuracy'],
        'threshold': best_threshold,
        'roc_ci': roc_ci
    }
    fi = open(fname, 'a')
    print(f'AUC of the overall Testing on {center_name} with {sampler} augmentation: {test_auc}',file=fi)
    print(f'Best AUC of PointEstimate in Bootstrap on {center_name} with {sampler} augmentation: {metrics_ci["auc"]},',file=fi)
    fi.close()
    figdata = [model_name, [test_proba, test_pred, y_test, test_auc, fpr, tpr, cm]]
    np.save(os.path.join(center_output_dir, f"{center_name}_{model_name}_{sampler}"), np.array(figdata, dtype=object))
    save_results(test_results, y_test, test_proba, test_pred, model_name, center_name, sampler, center_output_dir)
    return test_results, n_est, max_dep, best_threshold

def single_center_cv_adapt_ss(center_output_dir, l2ss, X_train, y_train, X_test, y_test, center_name, sampler, random_state=1, fname="a.txt"):
    model_name = 'SubspaceAligned'
    # Augmentation
    X_aug, y_aug = augmentation(center_name, X_train, y_train, sampler, dis_flag=False, random_state=random_state)
    # Training
    model = SubspaceAlignedClassifier(l2_regularization=l2ss, subspace_dim=15)
    model.fit(X_aug, y_aug, X_aug)

    train_proba = model.predict_proba(X_train)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_train, train_proba)
    youden_scores = tpr - fpr
    max_idx = np.argmax(youden_scores)
    best_threshold = thresholds[max_idx] if max_idx < len(thresholds) else 0.5

    # Testing
    test_proba = model.predict_proba(X_test)[:, 1]
    test_pred = (test_proba >= best_threshold).astype(int)
    fpr, tpr, thresholds = roc_curve(y_test, test_proba)
    cm = confusion_matrix(y_test, test_pred)

    test_auc = roc_auc_score(y_test, test_proba)
    test_sensitivity = recall_score(y_test, test_pred)
    test_specificity = recall_score(y_test, test_pred, pos_label=0)
    test_precision = precision_score(y_test, test_pred, zero_division=0)
    test_acc = accuracy_score(y_test, test_pred)

    # Bootstrap
    metrics_ci, roc_ci = bootstrap(model, X_test, y_test, best_threshold, n_bootstrap=1000)
    test_results = {
        'auc': metrics_ci['auc'],
        'sensitivity': metrics_ci['sensitivity'],
        'specificity': metrics_ci['specificity'],
        'precision': metrics_ci['precision'],
        'accuracy': metrics_ci['accuracy'],
        'threshold': best_threshold,
        'roc_ci': roc_ci
    }
    fi = open(fname, 'a')
    print(f'AUC of the overall Testing on {center_name} with {sampler} augmentation: {test_auc}',file=fi)
    print(f'Best AUC of PointEstimate in Bootstrap on {center_name} with {sampler} augmentation: {metrics_ci["auc"]},',file=fi)
    fi.close()
    figdata = [model_name, [test_proba, test_pred, y_test, test_auc, fpr, tpr, cm]]
    np.save(os.path.join(center_output_dir, f"{center_name}_{model_name}_{sampler}"), np.array(figdata, dtype=object))
    save_results(test_results, y_test, test_proba, test_pred, model_name, center_name, sampler, center_output_dir)
    return test_results, l2ss, best_threshold
