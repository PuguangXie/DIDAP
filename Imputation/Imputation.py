import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn import preprocessing
from Gain import gain, gain_test
from Utils import mad_loss, rmse_loss, generative_missdata
from Imputer import get_impute_mean, get_impute_tree, get_impute_zero, get_impute_knn_score, get_impute_iterative, get_impute_randforest
from Imputer import get_impute_multiple
from Imputer import final_impute_randforest
from Save_result import save_results
import missingno as msno
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

# ---------------- Imputation ---------------------
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer

################################# Imputation Comparison ###########################
#----------- Load and read data  -------------------
np.random.seed(20)
random_state=666
split=298
# hp_data = pd.read_excel('../Data/DATA_ALL_C0-C6.xlsx')
hp_data = pd.read_excel('../Data/raw_data.xlsx')
# feature_names = hp_data.columns[:-1].tolist()
data = hp_data.iloc[:, :]
# y = data[:, -1]
centers = {
    'DXH': data[:split],
    'FAH': data[split:]
}


for center_name, center_data in centers.items():
    print(f"\n========================= Processing data of {center_name} ===========================")
    center_data = center_data.iloc[:,-7:]
    # print(center_data.columns)
    X_miss_hip_fracture = center_data.iloc[:,:-1]
    y_miss_hip_fracture = center_data[X_miss_hip_fracture.notnull().T.all()]
    y_miss_hip_fracture = y_miss_hip_fracture.iloc[:, -1]
    X_miss_hip_fracture = X_miss_hip_fracture[X_miss_hip_fracture.notnull().T.all()]

    minmax = preprocessing.MinMaxScaler()
    X_miss_hip_fracture1 = minmax.fit_transform(X_miss_hip_fracture)
    X_miss_hip_fracture.iloc[:,:] = X_miss_hip_fracture1

    temp_miss_hip_fracture = X_miss_hip_fracture.copy()

    # k-fold cross validation
    from sklearn import datasets
    from sklearn.model_selection import KFold

    cv = 5
    kf = KFold(n_splits=cv, shuffle=True, random_state=1)  # 50 0.2 1
    miss = 0.11#0.001 0.003
    zero_mad = np.zeros(cv)
    kNN_mad = np.zeros(cv)
    MEAN_mad = np.zeros(cv)
    Iterative_mad = np.zeros(cv)
    Multiple_mad = np.zeros(cv)
    Multiple_mad_20 = np.zeros(cv)
    RandomForest_mad = np.zeros(cv)
    DecisionTree_mad = np.zeros(cv)
    Gan_mad = np.zeros(cv)

    zero_rmse = np.zeros(cv)
    kNN_rmse = np.zeros(cv)
    MEAN_rmse = np.zeros(cv)
    Iterative_rmse = np.zeros(cv)
    Multiple_rmse = np.zeros(cv)
    Multiple_rmse_20 = np.zeros(cv)
    RandomForest_rmse = np.zeros(cv)
    DecisionTree_rmse = np.zeros(cv)
    Gan_rmse = np.zeros(cv)
    number = 0
    # np.random.seed(20)
    for train_index, test_index in kf.split(X_miss_hip_fracture):  # k-fold
        # print('The', number, 'Fold： ')
        train_data = X_miss_hip_fracture.iloc[train_index]
        test = X_miss_hip_fracture.iloc[test_index]
        y_missing = y_miss_hip_fracture.copy()
        _, incomplete_data, _ = generative_missdata(test, miss_rate=miss)
        ori_data = X_miss_hip_fracture.copy()
        miss_data = X_miss_hip_fracture.copy()
        miss_data.iloc[test_index] = incomplete_data
        data_m = 1 - np.isnan(miss_data)
        data_m = np.array(data_m)
        #0-imputation
        zero_imputation = get_impute_zero(miss_data, ori_data, data_m)
        zero_mad[number], zero_rmse[number] = zero_imputation
        print('zero_rmse:', zero_imputation)
        # kNN-imputation
        kNN_imputation = get_impute_knn_score(miss_data, ori_data, data_m)
        kNN_mad[number], kNN_rmse[number] = kNN_imputation
        print('kNN_rmse:', kNN_imputation)
        # MEAN Imputation
        MEAN_imputation = get_impute_mean(miss_data, ori_data, data_m)
        MEAN_mad[number], MEAN_rmse[number] = MEAN_imputation
        print('MEAN_rmse:', MEAN_imputation)
        # Iterative imputation
        Iterative_imputation = get_impute_iterative(miss_data, ori_data, data_m)
        Iterative_mad[number], Iterative_rmse[number] = Iterative_imputation
        print('Iterative_rmse:', Iterative_imputation)
        # Multiple imputation (5)
        Multiple_imputation = get_impute_multiple(miss_data, ori_data, data_m, n_imputations=5)
        Multiple_mad[number], Multiple_rmse[number] = Multiple_imputation
        print('Multiple_rmse:', Multiple_imputation)
        # Multiple imputation (20)
        Multiple_imputation_20 = get_impute_multiple(miss_data, ori_data, data_m, n_imputations=20)
        Multiple_mad_20[number], Multiple_rmse_20[number] = Multiple_imputation_20
        print('Multiple(20)_rmse:', Multiple_imputation_20)
        # Random Forest Imputation
        RandomForest_imputation = get_impute_randforest(miss_data, ori_data, y_missing, data_m)
        RandomForest_mad[number], RandomForest_rmse[number] = RandomForest_imputation
        print('RandomForest_rmse:', RandomForest_imputation)
        # Decision Tree Imputation
        DecisionTree_imputation = get_impute_tree(miss_data, ori_data, y_missing, data_m)
        DecisionTree_mad[number], DecisionTree_rmse[number] = DecisionTree_imputation
        print('DecisionTree_rmse:', DecisionTree_imputation)
        # Gain
        gain_parameters = {'batch_size': 10,
                           'hint_rate': 0.95,
                           'alpha': 0.1,
                           'iterations': 1000}#50 0.95 0.1 20000
        imputed_data = gain(miss_data.values, gain_parameters, 1000)
        # imputed_data = gain_test(miss_data.values, 32)

        Gan_impute_mad = mad_loss(ori_data.values, imputed_data, data_m)
        Gan_impute_rmse = rmse_loss(ori_data.values, imputed_data, data_m)
        print('Gan_rmse:', Gan_impute_rmse)
        Gan_mad[number] = Gan_impute_mad
        Gan_rmse[number] = Gan_impute_rmse
        number = number + 1
        # if number>=1:
        #     break

    print("results-rmse:")
    print('zero_rmse:',np.mean(zero_rmse))
    print('kNN_rmse:',np.mean(kNN_rmse))
    print('MEAN_rmse:',np.mean(MEAN_rmse))
    print('Iterative_rmse:',np.mean(Iterative_rmse))
    print('Multiple_rmse:',np.mean(Multiple_rmse))
    print('Multiple(20)_rmse:', np.mean(Multiple_rmse_20))
    print('RandomForest_rmse:',np.mean(RandomForest_rmse))
    print('DecisionTree_rmse:',np.mean(DecisionTree_rmse))
    print('Gan_rmse:',np.mean(Gan_rmse))

    print("results-mad:")
    print('zero_mad:',np.mean(zero_mad))
    print('kNN_mad:',np.mean(kNN_mad))
    print('MEAN_mad:',np.mean(MEAN_mad))
    print('Iterative_mad:',np.mean(Iterative_mad))
    print('Multiple_mad:',np.mean(Multiple_mad))
    print('Multiple(20)_mad:', np.mean(Multiple_mad_20))
    print('RandomForest_mad:',np.mean(RandomForest_mad))
    print('DecisionTree_mad:',np.mean(DecisionTree_mad))
    print('Gan_mad:',np.mean(Gan_mad))

    mses_mean = []
    mads_mean = []
    mses_std = []
    mads_std = []
    mses_mean.append(np.mean(zero_rmse))
    mses_mean.append(np.mean(MEAN_rmse))
    mses_mean.append(np.mean(Iterative_rmse))
    mses_mean.append(np.mean(Multiple_rmse))
    mses_mean.append(np.mean(Multiple_rmse_20))
    mses_mean.append(np.mean(kNN_rmse))
    mses_mean.append(np.mean(RandomForest_rmse))
    mses_mean.append(np.mean(DecisionTree_rmse))
    mses_mean.append(np.mean(Gan_rmse))

    mses_std.append(np.std(zero_rmse))
    mses_std.append(np.std(MEAN_rmse))
    mses_std.append(np.std(Iterative_rmse))
    mses_std.append(np.std(Multiple_rmse))
    mses_std.append(np.std(Multiple_rmse_20))
    mses_std.append(np.std(kNN_rmse))
    mses_std.append(np.std(RandomForest_rmse))
    mses_std.append(np.std(DecisionTree_rmse))
    mses_std.append(np.std(Gan_rmse))

    mads_mean.append(np.mean(zero_mad))
    mads_mean.append(np.mean(MEAN_mad))
    mads_mean.append(np.mean(Iterative_mad))
    mads_mean.append(np.mean(Multiple_mad))
    mads_mean.append(np.mean(Multiple_mad_20))
    mads_mean.append(np.mean(kNN_mad))
    mads_mean.append(np.mean(RandomForest_mad))
    mads_mean.append(np.mean(DecisionTree_mad))
    mads_mean.append(np.mean(Gan_mad))

    mads_std.append(np.std(zero_mad))
    mads_std.append(np.std(MEAN_mad))
    mads_std.append(np.std(Iterative_mad))
    mads_std.append(np.std(Multiple_mad))
    mads_std.append(np.std(Multiple_mad_20))
    mads_std.append(np.std(kNN_mad))
    mads_std.append(np.std(RandomForest_mad))
    mads_std.append(np.std(DecisionTree_mad))
    mads_std.append(np.std(Gan_mad))
    print('mads_std:', mads_std)
    print('mses_std:', mses_std)
    save_results(mads_mean,mads_std, mses_mean,mses_std, center_name)


################################# FINAL Imputation ###########################

    print(f"Final imputation for {center_name} with {center_data.shape[0]} samples starts...")
    X_miss_hip_fracture = center_data.iloc[:,:-1]
    y_miss_hip_fracture = center_data.iloc[:, -1]

    X_miss_hip_fracture1 = minmax.fit_transform(X_miss_hip_fracture)
    X_miss_hip_fracture.iloc[:,:] = X_miss_hip_fracture1

    hp_data_re = center_data.copy()
    hp_data_re.iloc[:,:-1] = final_impute_randforest(X_miss_hip_fracture, y_miss_hip_fracture)
    writer = pd.ExcelWriter(f'./filled_{center_name}_data.xlsx')

    hp_data_re.to_excel(writer, sheet_name= 'Random Forest Imputation', index=False)
    writer._save()

    print('Final Imputation by Random Forest Imputer Done!')


