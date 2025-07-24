import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from libtlda.iw import ImportanceWeightedClassifier
from sklearn.metrics import (roc_curve, auc, confusion_matrix,
                             precision_score, recall_score, roc_auc_score)
from Evaluate_internal_center import single_center_cv, single_center_cv_adapt_iw, single_center_cv_adapt_ss
from Augmentation import augmentation
from libtlda.suba import SubspaceAlignedClassifier



def single_center_cv_with_param_search_iw(center_output_dir, X_train, y_train, X_test, y_test, center_name, sampler, random_state = 1, fname="a.txt"):
    l2_regularization = list(np.logspace(-3, 1, 20))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    results = []
    X_aug, y_aug = augmentation(center_name, X_train, y_train, sampler, dis_flag=False, random_state=random_state)
    for l2 in l2_regularization:
        all_thresholds = []
        fold_aucs = []
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_aug, y_aug)):
            model = ImportanceWeightedClassifier(l2_regularization=l2)
            X_train_fold, X_val = X_aug.iloc[train_idx], X_aug.iloc[val_idx]
            y_train_fold, y_val = y_aug.iloc[train_idx], y_aug.iloc[val_idx]
            model.fit(X_aug, y_aug, X_aug)

            fold_proba = model.predict_proba(X_train_fold)[:, 1]
            fpr, tpr, thresholds = roc_curve(y_train_fold, fold_proba)
            youden_idx = np.argmax(tpr - fpr)
            best_threshold = thresholds[youden_idx] if youden_idx < len(thresholds) else 0.5
            all_thresholds.append(best_threshold)

            val_proba = model.predict_proba(X_val)[:, 1]
            val_pred = (val_proba >= best_threshold).astype(int)

            pos_idx = np.where(y_val == 1)[0]
            fold_metrics = {
                'fold': fold,
                'threshold': best_threshold,
                'auc': roc_auc_score(y_val, val_proba),
                'sensitivity': recall_score(y_val, val_pred),
                'specificity': recall_score(y_val, val_pred, pos_label=0),
                'precision': precision_score(y_val, val_pred, zero_division=0)
            }
            fold_aucs.append(fold_metrics['auc'])
        mean_auc=np.mean(fold_aucs)
        results.append({
            'l2s': l2,
            'aucs': mean_auc
        })
    # Selecting the best hyperparameters
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('aucs', ascending=False)
    results_df.to_csv(os.path.join(center_output_dir, "param_search_results.csv"), index=False)
    fi = open(fname, 'a')
    print(f"{center_name} - Optimization finished and the best AUC: {results_df.iloc[0]['aucs']:.2f}",file=fi)
    print(f"Best parameters for Importance Weighted Transfer Learning is l2_regularization '{results_df.iloc[0]['l2s']}' ",file=fi)
    fi.close()
    best_l2 = results_df.iloc[0]['l2s']

    # Internal Testing
    test_results, l2, best_threshold = single_center_cv_adapt_iw(center_output_dir, best_l2, X_train, y_train, X_test, y_test, center_name, sampler, random_state=random_state, fname=fname)

    return test_results, l2, best_threshold


def single_center_cv_with_param_search_ss(center_output_dir, X_train, y_train, X_test, y_test, center_name, sampler, random_state=1, fname="a.txt"):
    l2_regularization = list(np.logspace(-3, 1, 20))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    results = []
    X_aug, y_aug = augmentation(center_name, X_train, y_train, sampler, dis_flag=False, random_state=random_state)
    for l2ss in l2_regularization:
        all_thresholds = []
        fold_aucs = []
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_aug, y_aug)):
            model = SubspaceAlignedClassifier(l2_regularization=l2ss)
            X_train_fold, X_val = X_aug.iloc[train_idx], X_aug.iloc[val_idx]
            y_train_fold, y_val = y_aug.iloc[train_idx], y_aug.iloc[val_idx]
            model.fit(X_aug, y_aug, X_aug)

            fold_proba = model.predict_proba(X_train_fold)[:, 1]
            fpr, tpr, thresholds = roc_curve(y_train_fold, fold_proba)
            youden_idx = np.argmax(tpr - fpr)
            best_threshold = thresholds[youden_idx] if youden_idx < len(thresholds) else 0.5
            all_thresholds.append(best_threshold)

            val_proba = model.predict_proba(X_val)[:, 1]
            val_pred = (val_proba >= best_threshold).astype(int)

            pos_idx = np.where(y_val == 1)[0]
            fold_metrics = {
                'fold': fold,
                'threshold': best_threshold,
                'auc': roc_auc_score(y_val, val_proba),
                'sensitivity': recall_score(y_val, val_pred),
                'specificity': recall_score(y_val, val_pred, pos_label=0),
                'precision': precision_score(y_val, val_pred, zero_division=0)
            }
            fold_aucs.append(fold_metrics['auc'])
        mean_auc=np.mean(fold_aucs)
        results.append({
            'l2ss': l2ss,
            'aucs': mean_auc
        })
    # Selecting the best hyperparameters
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('aucs', ascending=False)
    results_df.to_csv(os.path.join(center_output_dir, "param_search_results.csv"), index=False)
    fi = open(fname, 'a')
    print(f"{center_name} - Optimization finished and the best AUC: {results_df.iloc[0]['aucs']:.2f}",file=fi)
    print(f"Best parameters for Subspace Aligned Transfer Learning is l2_regularization '{results_df.iloc[0]['l2ss']}' ",file=fi)
    fi.close()

    best_l2 = results_df.iloc[0]['l2ss']

    # Internal Testing
    test_results, l2ss, best_threshold = single_center_cv_adapt_ss(center_output_dir, best_l2, X_train, y_train, X_test, y_test, center_name, sampler, random_state=random_state, fname=fname)

    return test_results, l2ss, best_threshold
