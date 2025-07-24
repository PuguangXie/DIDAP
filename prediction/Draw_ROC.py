import matplotlib.pyplot as plt
import numpy as np
import matplotlib

matplotlib.use('TkAgg')
centers= ['DXH','FAH']
ext_center_mapping = {
        'DXH': 'FAH',
        'FAH': 'DXH'
    }

for center in centers:
    print(center)
    ext = ext_center_mapping[center]

    lg_int = np.load(f'./Draw/{center}/{center}_LogisticRegression_None.npy',allow_pickle=True).tolist()
    lg_ext = np.load(f'./Draw/{center}/Int_{center}_Ext_{ext}_LogisticRegression_None.npy',allow_pickle=True).tolist()
    rf_int = np.load(f'./Draw/{center}/{center}_RandomForestClassifier_None.npy',allow_pickle=True).tolist()
    rf_ext = np.load(f'./Draw/{center}/Int_{center}_Ext_{ext}_RandomForestClassifier_None.npy',allow_pickle=True).tolist()
    svm_int = np.load(f'./Draw/{center}/{center}_SVC_None.npy',allow_pickle=True).tolist()
    svm_ext = np.load(f'./Draw/{center}/Int_{center}_Ext_{ext}_SVC_None.npy',allow_pickle=True).tolist()
    xgb_int = np.load(f'./Draw/{center}/{center}_XGBClassifier_None.npy',allow_pickle=True).tolist()
    xgb_ext = np.load(f'./Draw/{center}/Int_{center}_Ext_{ext}_XGBClassifier_None.npy',allow_pickle=True).tolist()
    iw_int = np.load(f'./Draw/{center}/{center}_ImportanceWeighted_None.npy',allow_pickle=True).tolist()
    iw_ext = np.load(f'./Draw/{center}/Int_{center}_Ext_{ext}_ImportanceWeighted_None.npy',allow_pickle=True).tolist()
    ssa_int = np.load(f'./Draw/{center}/{center}_SubspaceAligned_None.npy',allow_pickle=True).tolist()
    ssa_ext = np.load(f'./Draw/{center}/Int_{center}_Ext_{ext}_SubspaceAligned_None.npy',allow_pickle=True).tolist()

    full_fig_dataset=[]
    full_fig_dataset.append(lg_int)
    full_fig_dataset.append(lg_ext)
    full_fig_dataset.append(rf_int)
    full_fig_dataset.append(rf_ext)
    full_fig_dataset.append(svm_int)
    full_fig_dataset.append(svm_ext)
    full_fig_dataset.append(xgb_int)
    full_fig_dataset.append(xgb_ext)
    full_fig_dataset.append(iw_int)
    full_fig_dataset.append(iw_ext)
    full_fig_dataset.append(ssa_int)
    full_fig_dataset.append(ssa_ext)

    # figdata = [model_name, [test_proba, test_pred, y_test, test_auc, fpr, tpr, cm]]
    model_colors = {
        'ImportanceWeighted': '#1f77b4',  # blue
        'LogisticRegression': '#ff7f0e',  # orange
        'RandomForestClassifier': '#2ca02c',  # green
        'SubspaceAligned': '#d62728',  # red
        'SVC': '#9467bd',  # purple
        'XGBClassifier': '#8c564b'  # brown
    }
    model_mapping ={
        'ImportanceWeighted': 'ImportanceWeighted',
        'LogisticRegression': 'LogisticRegression',
        'RandomForestClassifier': 'RandomForest',
        'SubspaceAligned': 'SubspaceAligned',
        'SVC': 'SVM',
        'XGBClassifier': 'XGBoost'
    }
    figidx = 0
    figjdx = 0
    for i, (model_name, data) in enumerate(full_fig_dataset):
        fpr = data[4]
        tpr = data[5]
        test_auc = data[3]
        cm = data[6]
        location = 'Internal' if i % 2 == 0 else 'External'
        if location == 'Internal':
            # ROC: internal
            plt.figure(num=1, figsize=(7, 7), dpi=100)
            plt.plot(fpr, tpr, lw=1, label='{} (Internal: AUC={:.3f})'.format(model_mapping[model_name], test_auc), color=model_colors[model_name])
            plt.plot([0, 1], [0, 1], '--', lw=2, color='grey')
            plt.axis('square')
            plt.xlim([-0.01, 1])
            plt.ylim([-0.01, 1.02])
            plt.xlabel('False Positive Rate', fontsize=15)
            plt.ylabel('True Positive Rate', fontsize=15)
            plt.title(f'ROC Curve in {center}', fontsize=15)
            plt.legend(loc='lower right', fontsize=8)

            # confusion_matrix: internal
            plt.figure(num=2, figsize=(8, 6))
            plt.suptitle('Internal Validation Confusion Matrix', fontsize=17)
            ax = plt.subplot(2, 3, figidx + 1)
            ax.matshow(cm, cmap="Blues")
            thresh = cm.max() / 1.5
            for i, j in zip([0, 0, 1, 1], [0, 1, 0, 1]):
                ax.text(j, i, '{:.2f}'.format(cm[i, j]), horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black", fontsize=13)
            ax.set_title(model_mapping[model_name], fontsize=13)
            ax.set_ylabel('True Label', fontsize=13)
            ax.set_xlabel('Predicted Label', fontsize=13)
            plt.subplots_adjust(left=0.1, right=0.95, bottom=0.05, top=0.85, wspace=0.5, hspace=0.5)
            figidx = figidx + 1

        if location == 'External':
            # ROC: external
            plt.figure(num=1, figsize=(7, 7), dpi=100)
            plt.plot(fpr, tpr, '--', lw=1, label='{} (External: AUC={:.3f})'.format(model_mapping[model_name], test_auc), color=model_colors[model_name])
            plt.legend(loc='lower right', fontsize=8)

            # confusion_matrix: external
            plt.figure(num=3, figsize=(8, 6))
            plt.suptitle('External Validation Confusion Matrix', fontsize=17)
            ax = plt.subplot(2, 3, figjdx + 1)
            ax.matshow(cm, cmap="Blues")
            thresh = cm.max() / 1.5
            for i, j in zip([0, 0, 1, 1], [0, 1, 0, 1]):
                ax.text(j, i, '{:.2f}'.format(cm[i, j]), horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black", fontsize=13)
            ax.set_title(model_mapping[model_name], fontsize=13)
            ax.set_ylabel('True Label', fontsize=13)
            ax.set_xlabel('Predicted Label', fontsize=13)
            plt.subplots_adjust(left=0.1, right=0.95, bottom=0.05, top=0.85, wspace=0.5, hspace=0.5)

            figjdx = figjdx + 1

    plt.figure(num=1).savefig(center + '_roc_curve.svg')
    plt.figure(num=2).savefig(center + '_cm_internal.svg')
    plt.figure(num=3).savefig(center + '_cm_external.svg')

