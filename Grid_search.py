import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn import preprocessing, clone
from sklearn.model_selection import GridSearchCV
import seaborn as sns
from Save_result import save_results, save_best_model
from Model import get_model_param_grid
from Evaluate_internal_center import single_center_cv
from Augmentation import augmentation
import time

def grid_search_hyperparam_optimization(center_name, model, X_train, y_train, param_grid, sampler, cv=5, random_state=1,fname="a.txt"):
    model_name = type(model).__name__

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring='roc_auc',
        cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state),
        verbose=1,
        refit=True
    )
    # dis_flag: Whether to do sample distribution analysis
    # Augmentation
    X_aug, y_aug = augmentation(center_name, X_train, y_train, sampler, dis_flag=False)

    start_time = time.time()
    fi = open(fname, 'a')
    print(f"Grid Search Starts: {time.strftime('%Y-%m-%d %H:%M:%S')}",file=fi)
    fi.close()

    grid_search.fit(X_aug, y_aug)

    end_time = time.time()
    fi = open(fname, 'a')
    print(f"Grid Search Ends: {time.strftime('%Y-%m-%d %H:%M:%S')}",file=fi)
    print(f"Time costs: {end_time - start_time:.2f}seconds.",file=fi)
    fi.close()

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    fi = open(fname, 'a')
    print(f"Best parameters: {best_params}",file=fi)
    print(f"Best AUC: {best_score:.4f}",file=fi)
    fi.close()

    results_df = pd.DataFrame(grid_search.cv_results_)
    results_df = results_df.sort_values('rank_test_score')
    keep_columns = ['params', 'mean_test_score', 'std_test_score', 'rank_test_score']
    for param in param_grid.keys():
        param = f"param_{param}"
        if param not in keep_columns:
            keep_columns.append(param)
    results_df = results_df[keep_columns]
    return best_model, best_params, best_score, results_df


def single_center_cv_with_grid_search(center_output_dir, model, X_train, y_train, X_test, y_test, center_name, sampler, optimize_params=True, random_state=1, fname="a.txt"):

    # global best_params
    model_name = type(model).__name__

    if optimize_params:
        fi = open(fname, 'a')
        print(f"\n==== {center_name} - {model_name} Optimization starts ====",file=fi)
        fi.close()
        param_grid = get_model_param_grid(model_name)
        optimized_model, best_params, best_score, search_results = grid_search_hyperparam_optimization(center_name, model, X_train, y_train, param_grid, sampler, cv=5, random_state=random_state,fname=fname)
        save_best_model(optimized_model, center_output_dir, model_name, sampler)

        search_results.to_csv(os.path.join(center_output_dir, "grid_search_results.csv"), index=False)
        pd.DataFrame([best_params]).to_csv(os.path.join(center_output_dir, "best_params.csv"), index=False)
        fi = open(fname, 'a')
        print(f"{center_name} - Optimization finished and the best AUC: {best_score:.4f}",file=fi)
        fi.close()
        generate_grid_search_report(search_results, sampler, center_output_dir)
    else:
        print(f"\n==== {center_name} Default model ====")
        optimized_model = clone(model)
    fi = open(fname, 'a')
    print(f"\n==== {center_name} - {model_name} Best model selected, training starts ====",file=fi)
    fi.close()
    test_results, final_model, best_threshold = single_center_cv(center_output_dir, optimized_model, X_train, y_train, X_test, y_test, center_name, sampler, random_state=random_state,fname=fname)

    return test_results, final_model, best_threshold, best_params


def generate_grid_search_report(search_results, sampler, output_dir):

    pdf_path = os.path.join(output_dir, f"{sampler}_grid_search_report.pdf")

    with PdfPages(pdf_path) as pdf:
        param_names = [col for col in search_results.columns if
                       col not in ['params', 'mean_test_score', 'std_test_score', 'rank_test_score']]
        plt.figure(figsize=(14, 10))
        n_params = len(param_names)
        n_cols = min(2, n_params)
        n_rows = (n_params + n_cols - 1) // n_cols

        for i, param in enumerate(param_names):
            plt.subplot(n_rows, n_cols, i + 1)

            if search_results[param].dtype == 'object' or len(search_results[param].unique()) < 5:
                sns.boxplot(x=param, y='mean_test_score', data=search_results)
            else:
                plt.scatter(search_results[param], search_results['mean_test_score'])
                plt.xlabel(param)
                plt.ylabel('AUC')

            plt.title(f"Parameters: {param}")

        plt.suptitle("Significance of parameters", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig()
        plt.close()

        plt.figure(figsize=(12, 8))
        plt.axis('off')
        top_results = search_results.head(10)
        report_text = "Top 10 AUC parameters:\n\n"
        for i, (_, row) in enumerate(top_results.iterrows()):
            report_text += f"Ranking #{i + 1} (AUC={row['mean_test_score']:.4f}):\n"
            for param in param_names:
                report_text += f"  {param}: {row[param]}\n"
            report_text += "\n"

        plt.text(0.5, 0.5, report_text, ha='center', va='center', fontsize=10)
        plt.title('Best parameters', fontsize=16)
        pdf.savefig()
        plt.close()

    print(f"Grid Search Report with {sampler} Augmentation Saved: {pdf_path}")
