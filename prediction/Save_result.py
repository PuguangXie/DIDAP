import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import (roc_curve, auc, confusion_matrix,
                             precision_score, recall_score, roc_auc_score)
import seaborn as sns
import joblib


def save_results(test_results, test_true, test_proba, test_pred, model_name, center_name, sampler, output_dir):
    pdf_path = os.path.join(output_dir, f"{center_name}_{model_name}_{sampler} augmentation_report.pdf")
    with PdfPages(pdf_path) as pdf:
        plt.figure(figsize=(10, 6))
        plt.axis('off')
        test_text = "\n".join([
            f"\nResults of Testing dataset (95% CI):",
            f"  AUC: {test_results['auc']['point_estimate']:.4f} ({test_results['auc']['ci_lower']:.4f}-{test_results['auc']['ci_upper']:.4f})",
            f"  Sensitivity: {test_results['sensitivity']['point_estimate']:.4f} ({test_results['sensitivity']['ci_lower']:.4f}-{test_results['sensitivity']['ci_upper']:.4f})",
            f"  Specificity: {test_results['specificity']['point_estimate']:.4f} ({test_results['specificity']['ci_lower']:.4f}-{test_results['specificity']['ci_upper']:.4f})",
            f"  Precision: {test_results['precision']['point_estimate']:.4f} ({test_results['precision']['ci_lower']:.4f}-{test_results['precision']['ci_upper']:.4f})",
            f"  Threshold: {test_results['threshold']:.4f}"
        ])

        plt.text(0.5, 0.5, f"{model_name} Performance\n\n{test_text}",
                 ha='center', va='center', fontsize=12)
        plt.title(f"{center_name} - {model_name} -{sampler} Augmentation Performance Report", fontsize=16)
        pdf.savefig()
        plt.close()
        if len(test_true) > 0 and len(np.unique(test_true)) >= 2:
            plt.figure(figsize=(10, 8))
            # point estimate
            fpr_point, tpr_point, _ = roc_curve(test_true, test_proba)
            roc_auc_point = auc(fpr_point, tpr_point)
            plt.plot(fpr_point, tpr_point, color='blue', lw=2, alpha=0.8, label=f'ROC Curve (AUC={roc_auc_point:.4f})')

            plt.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.8)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('FPR')
            plt.ylabel('TPR')
            plt.title(f'{center_name} - Testing ROC Curve')
            plt.legend(loc='lower right')
            plt.grid(True, alpha=0.3)
            pdf.savefig()
            plt.close()

        overall_cm = confusion_matrix(test_true, test_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(overall_cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Pre_0', 'Pred_1'],
                    yticklabels=['Label_0', 'Label_1'])
        plt.xlabel('Prediction')
        plt.ylabel('True Label')
        plt.title(f'{center_name} - Overall Confusion Matrix\n(Sum of {test_true.value_counts().get(1, 0)} deceased cases, and {test_true.value_counts().get(0, 0)} negative samples)')
        pdf.savefig()
        plt.close()

    results_df = pd.DataFrame({
        'metric': ['auc', 'sensitivity', 'specificity', 'precision', 'threshold'],
        'test_point_estimate': [
            test_results['auc']['point_estimate'],
            test_results['sensitivity']['point_estimate'],
            test_results['specificity']['point_estimate'],
            test_results['precision']['point_estimate'],
            test_results['threshold']
        ],
        'test_ci_lower': [
            test_results['auc']['ci_lower'],
            test_results['sensitivity']['ci_lower'],
            test_results['specificity']['ci_lower'],
            test_results['precision']['ci_lower'],
            np.nan
        ],
        'test_ci_upper': [
            test_results['auc']['ci_upper'],
            test_results['sensitivity']['ci_upper'],
            test_results['specificity']['ci_upper'],
            test_results['precision']['ci_upper'],
            np.nan
        ]
    })
    results_df.to_csv(os.path.join(output_dir, f"{sampler}_performance_results.csv"), index=False)


def save_best_model(model, output_dir, model_name, sampler):
    # os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, f"Best_{model_name}_model_{sampler}.joblib")
    joblib.dump(model, model_path)
    print(f"Best model with {sampler} augmentation saved: {model_path}")
    return model_path

