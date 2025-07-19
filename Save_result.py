import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import (roc_curve, auc, confusion_matrix,
                             precision_score, recall_score, roc_auc_score)
import seaborn as sns


def save_results(mads_mean,mads_std, mses_mean, mses_std, center_name):
    pdf_path = os.path.join(f"{center_name} Imputation Comparison Report.pdf")
    with PdfPages(pdf_path) as pdf:
        plt.figure(figsize=(18, 10))
        # plt.axis('off')
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams.update({'font.size': 10})
        # plt.rcParams.update({'font.size':15})

        x_labels = ['Zero Imputation', 'Mean Imputation', f'Iterative Imputation\n(1 Imputation)', f'Multiple Imputation\n(5 Imputations)', f'Multiple Imputation\n(20 Imputations)',
                    'KNN Imputation', 'RandomForest Imputation', 'DecisionTree Imputation', 'Gain Imputation']
        n_bars = len(mses_mean)
        xval = np.arange(n_bars)
        colors = ["r", "g", "b", "gray", "pink", "skyblue", "orange", "black", "c"]
        sns.set(style='dark')
        matplotlib.use('TkAgg')

        for j in xval:
            plt.barh(
                j - 0.2,
                mads_mean[j],
                xerr=mads_std[j],
                color=colors[j],
                alpha=0.6,
                align="center",
                capsize=3
            )
            plt.text(mads_mean[j] + 0.01, j, '%s' % "%.4f" % mads_mean[j], va='top', fontsize=15)  # 18
        plt.subplots_adjust(left=0.3)
        # pdf.savefig(fig, bbox_inches='tight')
        plt.title(f"Imputation Techniques Comparison in {center_name} with Hip Fracture Data (MAD)", fontsize=26)  # 12
        plt.xlim(left=np.min(mads_mean) * 0., right=np.max(mads_mean) * 1.7)
        # plt.yticks(xval)
        plt.yticks(xval, x_labels, fontsize=16)
        plt.xlabel("MAD", fontsize=18)
        plt.savefig(f'Imputation Techniques Comparison in {center_name} with Hip Fracture Data (MAD).tiff', dpi=300, format='tiff')
        plt.show()
        plt.close()

        plt.figure(figsize=(18, 10))
        for j in xval:
            plt.barh(
                j - 0.2,
                mses_mean[j],
                xerr=mses_std[j],
                color=colors[j],
                alpha=0.6,
                align="center",
                capsize=3
            )
            plt.text(mses_mean[j] + 0.01, j, '%s' % "%.4f" % mses_mean[j], va='top', fontsize=15)  # 18
        plt.subplots_adjust(left=0.3)
        # pdf.savefig(fig, bbox_inches='tight')
        plt.title(f"Imputation Techniques Comparison in {center_name} with Hip Fracture Data (RMSE)", fontsize=26)  # 12
        plt.xlim(left=np.min(mses_mean) * 0., right=np.max(mses_mean) * 1.7)
        # plt.yticks(xval)
        plt.yticks(xval, x_labels, fontsize=16)
        plt.xlabel("RMSE", fontsize=18)
        plt.savefig(f'Imputation Techniques Comparison in {center_name} with Hip Fracture Data (RMSE).tiff', dpi=300, format='tiff')

        plt.show()
        plt.close()
