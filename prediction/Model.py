import numpy as np


def get_model_param_grid(model_name):
    if model_name == "XGBClassifier":
        return {
            'max_depth': list(range(1, 5, 1)),
            'learning_rate': list(np.arange(0.05, 0.5, 0.05)),
            'reg_alpha': list(np.arange(0.1, 1, 0.1)),
            'reg_lambda': list(np.arange(0.1, 1, 0.1))
        }
    elif model_name == "RandomForestClassifier":
        return {
            'n_estimators': list(range(10, 200, 10)),
            'max_depth': list(range(1, 8, 1))
        }
    elif model_name == "LogisticRegression":
        return {
            'C': list(np.arange(0.05, 2, 0.05))
        }
    elif model_name == "SVC":
        return {
            'C': list(np.arange(500, 1000, 100))
        }
    # elif model_name == "ImportanceWeighted":
    #     return {
    #         'n_estimators': list(range(10, 200, 10)),
    #         'max_depth': list(range(1, 8, 1))
    #     }
    # elif model_name == "SubspaceAligned":
    #     return {
    #         'l2_regularization': list(np.arange(0.5, 1, 0.5))
    #     }
    else:
        return {}
