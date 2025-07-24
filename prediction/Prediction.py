import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing, clone
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from Grid_search import single_center_cv_with_grid_search
from Evaluate_external_center import evaluate_on_other_center, evaluate_on_other_center_adapt_iw, evaluate_on_other_center_adapt_ss
from Evaluate_internal_center import single_center_cv_adapt_iw, single_center_cv_adapt_ss
from libtlda.iw import ImportanceWeightedClassifier
from libtlda.suba import SubspaceAlignedClassifier
import warnings
warnings.filterwarnings("ignore")

def load_and_preprocess_data(file_path, split):
    hp_data = pd.read_excel(file_path)
    # print(hp_data.shape)
    feature_names = hp_data.columns[:-1].tolist()
    data = np.array(hp_data.iloc[:, :])
    y = data[:, -1]
    centers = {
        'DXH': {
            'raw_X': data[:split][:, :-1],
            'y': y[:split]
        },
        'FAH': {
            'raw_X': data[split:][:, :-1],
            'y': y[split:]
        }
    }
    for center_name, center_data in centers.items():
        minmax = preprocessing.MinMaxScaler()
        X_scaled = minmax.fit_transform(center_data['raw_X'])

        center_df = pd.DataFrame(X_scaled, columns=feature_names)
        center_df['Result'] = center_data['y'].astype(int)
        center_df['Center'] = center_name
        centers[center_name]['df'] = center_df

    return centers

def format_metric(point, lower, upper):
    return f"{point:.2f}\n[{lower:.2f}, {upper:.2f}]"


if __name__ == "__main__":
    # Augmentation: choosing a specific sampler: smote, ros, None
    # augmentation_methods = ['None','smote','ros']     # Comparison of augmentation methods
    augmentation_methods = ['None']     # Prediction with none augmentation
    random_state = 200
    np.random.seed(random_state)
    fname = 'result.txt'
    fi = open(fname, 'w')
    fi.close()
    centers = load_and_preprocess_data('../Data/fill_data_final.xlsx', split=298)
    model_configs = [
        {
            'name': 'LogisticRegression',
            'model': LogisticRegression(C=0.1, max_iter=1000, random_state=random_state),
            'description': 'Logistic Regression Classifier',
            'is_transfer': False
        },{
            'name': 'RandomForest',
            'model': RandomForestClassifier(n_estimators=200, max_depth=2, random_state=random_state),
            'description': 'Random Forest Classifier',
            'is_transfer': False
        },
        {
            'name': 'SVM',
            'model': svm.SVC(C=0.3, kernel='rbf', gamma='scale', probability=True, random_state=random_state),
            'description': 'SVM Classifier',
            'is_transfer': False
        }
        ,{
            'name': 'XGBoost',
            'model': XGBClassifier(eval_metric='logloss', random_state=random_state),
            'description': 'XGBoost Classifier',
            'is_transfer': False
        },
        {
            'name': 'ImportanceWeighted',
            'model': ImportanceWeightedClassifier(weight_estimator='lr', loss_function= 'gini', n_estimators= 50, max_depth = 1, kernel_type='rbf'),
            'description': 'Importance Weighted Transfer Learning',
            'is_transfer': True
        },
        {
            'name': 'SubspaceAligned',
            'model': SubspaceAlignedClassifier(l2_regularization=0.01),
            'description': 'Subspace Aligned Transfer Learning',
            'is_transfer': True
        }
    ]
    lg_l2 = 0
    n_est = 0
    max_dep = 0
    center_data_splits = {}
    ext_center_mapping = {
        'DXH': 'FAH',
        'FAH': 'DXH'
    }

    for center_name, center_data in centers.items():
        data = center_data['df']
        X = data.drop(['Result', 'Center'], axis=1)
        y = data['Result']
        fi = open(fname, 'a')
        print(f"{center_name}: {data.shape[0]} cases | Deceased cases: {sum(y == 1)} ({sum(y == 1) / len(y):.1%})",file=fi)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=random_state)
        print(f"{center_name}\n Training + samples: {len(X_train.iloc[np.where(y_train == 1)[0]])} | - samples: {len(X_train.iloc[np.where(y_train == 0)[0]])} \n Testing + samples: {len(X_test.iloc[np.where(y_test == 1)[0]])} | - samples: {len(X_test.iloc[np.where(y_test == 0)[0]])} ",file=fi)
        fi.close()
        center_data_splits[center_name] = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test
        }

    for sampler in augmentation_methods:
        fi = open(fname, 'a')
        print(sampler,file=fi)
        fi.close()
        for center_name in centers:
            fi = open(fname, 'a')
            print(f"\n========================= Processing data of {center_name} with {sampler} ========================",file=fi)
            print(f"\n========================= Processing data of {center_name} with {sampler} ========================")
            fi.close()
            int_data = center_data_splits[center_name]
            X_train = int_data['X_train']
            y_train = int_data['y_train']
            X_test = int_data['X_test']
            y_test = int_data['y_test']
            for model_config in model_configs:
                if model_config['name'] == 'ImportanceWeighted':
                    base_model = ImportanceWeightedClassifier(weight_estimator='lr', l2_regularization=0.01, kernel_type='rbf')
                elif model_config['name'] == 'SubspaceAligned':
                   base_model = SubspaceAlignedClassifier(l2_regularization=0.01)
                else:
                    base_model = clone(model_config['model'])
                center_output_dir = f"{sampler}_{center_name}_{model_config['name']}_results"
                os.makedirs(center_output_dir, exist_ok=True)
                # performance of predicting testing dataset on the internal center
                if model_config['name'] == 'ImportanceWeighted':
                    test_results, n_est, max_dep, threshold_iw = single_center_cv_adapt_iw(center_output_dir, n_est, max_dep, X_train, y_train, X_test, y_test, center_name, sampler, random_state=random_state, fname=fname)
                elif model_config['name'] == 'SubspaceAligned':
                    test_results, l2ss, threshold_ss = single_center_cv_adapt_ss(center_output_dir, lg_l2, X_train, y_train, X_test, y_test, center_name, sampler, random_state=random_state, fname=fname)
                else:
                    test_results, final_model, best_threshold, best_params = single_center_cv_with_grid_search(center_output_dir, base_model, X_train, y_train, X_test, y_test, center_name, sampler, optimize_params=True, random_state=random_state, fname=fname)
                if model_config['name'] == 'LogisticRegression':
                    lg_l2 = best_params['C']
                if model_config['name'] == 'RandomForest':
                    n_est = best_params['n_estimators']
                    max_dep = best_params['max_depth']
                # performance of predicting testing dataset on the external center
                ext_center = ext_center_mapping[center_name]
                ext_data = center_data_splits[ext_center]
                X_train_ext = ext_data['X_train']
                y_train_ext = ext_data['y_train']
                X_test_ext = ext_data['X_test']
                y_test_ext = ext_data['y_test']
                if model_config['name'] == 'ImportanceWeighted':
                    metrics = evaluate_on_other_center_adapt_iw(n_est, max_dep, threshold_iw, X_train, y_train, X_train_ext, X_test_ext, y_test_ext, center_name, ext_center, sampler,center_output_dir,fname=fname)
                elif model_config['name'] == 'SubspaceAligned':
                    print(lg_l2)
                    metrics = evaluate_on_other_center_adapt_ss(lg_l2, threshold_ss, X_train, y_train, X_train_ext, X_test_ext, y_test_ext, center_name, ext_center, sampler,center_output_dir,fname=fname)
                else:
                    metrics = evaluate_on_other_center(final_model, X_test_ext, y_test_ext, center_name, ext_center, best_threshold, sampler,center_output_dir,fname=fname)

                results_df = pd.DataFrame({
                    'model': [model_config['name']],
                    'center': [center_name],
                    'auc':  format_metric(test_results['auc']['point_estimate'], test_results['auc']['ci_lower'],test_results['auc']['ci_upper']),
                    'accuracy': format_metric(test_results['accuracy']['point_estimate'], test_results['accuracy']['ci_lower'], test_results['accuracy']['ci_upper']),
                    'sensitivity': format_metric(test_results['sensitivity']['point_estimate'],test_results['sensitivity']['ci_lower'], test_results['sensitivity']['ci_upper']),
                    'specificity': format_metric(test_results['specificity']['point_estimate'], test_results['specificity']['ci_lower'],test_results['specificity']['ci_upper']),
                    'ext_auc': format_metric(metrics['auc']['point_estimate'], metrics['auc']['ci_lower'],metrics['auc']['ci_upper']),
                    'ext_accuracy': format_metric(metrics['accuracy']['point_estimate'], metrics['accuracy']['ci_lower'], metrics['accuracy']['ci_upper']),
                    'ext_sensitivity': format_metric(metrics['sensitivity']['point_estimate'],metrics['sensitivity']['ci_lower'], metrics['sensitivity']['ci_upper']),
                    'ext_specificity': format_metric(metrics['specificity']['point_estimate'], metrics['specificity']['ci_lower'],metrics['specificity']['ci_upper'])
                    })

                all_results_file = os.path.join(f"{random_state}/", f"All_Models_{sampler}_Performance_datasize.csv")
                if os.path.exists(all_results_file):
                    results_df.to_csv(all_results_file, mode='a', header=False, index=False)
                else:
                    results_df.to_csv(all_results_file, index=False)
                fi = open(fname, 'a')
                print(f"{model_config['description']} evaluation completed for {center_name} and external center of {ext_center}", file=fi)
                fi.close()

