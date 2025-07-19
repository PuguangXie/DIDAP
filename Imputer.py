import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn import preprocessing
from Utils import mad_loss, rmse_loss, generative_missdata

import warnings
warnings.filterwarnings("ignore")

# ---------------- Imputation ---------------------
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer



# Multiple Imputation
def get_impute_multiple(X_missing, ori_data, data_m, n_imputations):
    imputed_datasets = []
    for i in range(n_imputations):
        imputer = IterativeImputer(
            missing_values=np.nan,
            add_indicator=False,
            n_nearest_features=5,
            sample_posterior=True,
            initial_strategy='mean',
            min_value=0,
            random_state=i
        )
        imputed_data = imputer.fit_transform(X_missing)
        imputed_datasets.append(imputed_data)

    avg_imputed = np.mean(imputed_datasets, axis=0)

    data1 = ori_data.values
    data2 = avg_imputed
    multi_mad = mad_loss(data1, data2, data_m)
    multi_rmse = rmse_loss(data1, data2, data_m)

    return multi_mad, multi_rmse


#0-imputation
def get_impute_zero(X_missing, ori_data, data_m):

    imputer = SimpleImputer(
        missing_values=np.nan, add_indicator=False, strategy="constant", fill_value=0
    )
    # hp_data_re = hp_data.copy()
    new_x = imputer.fit_transform(X_missing)
    data1 = ori_data.values
    data2 = new_x
    # data1 = renormalization(data1, norm_parameters)
    # data2 = renormalization(data2, norm_parameters)
    zero_impute_mad = mad_loss(data1, data2, data_m)
    zero_impute_rmse = rmse_loss(data1, data2,data_m)
    return zero_impute_mad, zero_impute_rmse

# kNN-imputation
def get_impute_knn_score(X_missing, ori_data, data_m):
    imputer = KNNImputer(missing_values=np.nan, add_indicator=False)
    new_x = imputer.fit_transform(X_missing)
    # hp_data_re = hp_data.copy()
    # hp_data_re.iloc[:,start:end] = new_x
    # writer = pd.ExcelWriter('E:/FINAL/data_all_imputer.xlsx', engin='openpyxl')
    # book = load_workbook(writer.path)
    # writer.book = book
    # hp_data_re.to_excel(writer, sheet_name= 'KNN Imputation', index=False)
    # writer.save()
    data1 = ori_data.values
    data2 = new_x
    # data1 = renormalization(data1, norm_parameters)
    # data2 = renormalization(data2, norm_parameters)
    knn_impute_mad = mad_loss(data1, data2,data_m)
    knn_impute_rmse = rmse_loss(data1, data2,data_m)
    return knn_impute_mad, knn_impute_rmse

# MEAN Imputation
def get_impute_mean(X_missing, ori_data, data_m):
    imputer = SimpleImputer(missing_values=np.nan, strategy="mean", add_indicator=False)
    new_x = imputer.fit_transform(X_missing)
    xx = np.nanmean(np.array(X_missing), axis = 0)
    # hp_data_re = hp_data.copy()
    # hp_data_re.iloc[:,start:end] = new_x
    # writer = pd.ExcelWriter('E:/FINAL/data_all_imputer.xlsx', engin='openpyxl')
    # book = load_workbook(writer.path)
    # writer.book = book
    # hp_data_re.to_excel(writer, sheet_name= 'Mean Imputation', index=False)
    # writer.save()
    data1 = ori_data.values
    data2 = new_x
    # data1 = renormalization(data1, norm_parameters)
    # data2 = renormalization(data2, norm_parameters)
    mean_impute_mad = mad_loss(data1, data2, data_m)
    mean_impute_rmse = rmse_loss(data1, data2,data_m)
    return mean_impute_mad, mean_impute_rmse

# Iterative imputation
def get_impute_iterative(X_missing, ori_data, data_m):
    imputer = IterativeImputer(
        missing_values=np.nan,
        add_indicator=False,
        random_state=0,
        n_nearest_features=5,
        sample_posterior=True,
        initial_strategy='mean',
        min_value=0,
    )
    new_x = imputer.fit_transform(X_missing)
    # hp_data_re = hp_data.copy()
    # hp_data_re.iloc[:,start:end] = new_x
    # writer = pd.ExcelWriter('E:/FINAL/data_all_imputer.xlsx', engin='openpyxl')
    # book = load_workbook(writer.path)
    # writer.book = book
    # hp_data_re.to_excel(writer, sheet_name= 'Iterative Imputation', index=False)
    # writer.save()
    data1 = ori_data.values
    data2 = new_x
    # data1 = renormalization(data1, norm_parameters)
    # data2 = renormalization(data2, norm_parameters)
    iterative_impute_mad = mad_loss(data1, data2, data_m)
    iterative_impute_rmse = rmse_loss(data1, data2,data_m)
    return iterative_impute_mad, iterative_impute_rmse

#Random Forest Imputation
def get_impute_randforest(X_missing, ori_data, y_missing, data_m):
    temp = pd.DataFrame(X_missing.values, columns=X_missing.columns.astype(str).tolist())
    X_missing = temp
    y_missing = np.array(y_missing)
    # temp.iloc[:,:] = X_missing.iloc[:,:]
    X_missing_reg = X_missing.copy()
    X_df = X_missing_reg.isnull().sum()
    colname = X_df[~X_df.isin([0])].sort_values().index.values
    sortindex = []
    for i in colname:
        sortindex.append(X_missing_reg.columns.tolist().index(str(i)))

    for i in sortindex:
        df = X_missing_reg
        fillc = df.iloc[:, i]

        df = pd.concat([df.drop(df.columns[i], axis=1), pd.DataFrame(y_missing, columns=['y_missing'])], axis=1)

        df_0 = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0).fit_transform(df)

        Ytrain = fillc[fillc.notnull()]
        Ytest = fillc[fillc.isnull()]

        Xtrain = df_0[Ytrain.index, :]
        Xtest = df_0[Ytest.index, :]

        rfc = RandomForestRegressor(random_state=2)
        rfc = rfc.fit(Xtrain, Ytrain)
        Ypredict = rfc.predict(Xtest)

        X_missing_reg.loc[X_missing_reg.iloc[:, i].isnull(), X_missing_reg.columns[i]] = Ypredict
    # hp_data_re = hp_data.copy()
    # hp_data_re.iloc[:,start:end] = X_missing_reg
    # writer = pd.ExcelWriter('E:/FINAL/data_all_imputer.xlsx', engin='openpyxl')
    # book = load_workbook(writer.path)
    # writer.book = book
    # hp_data_re.to_excel(writer, sheet_name= 'Randforest Imputation', index=False)
    # writer.save()
    data1 = ori_data.values
    data2 = X_missing_reg.values
    # data1 = renormalization(data1, norm_parameters)
    # data2 = renormalization(data2, norm_parameters)
    randforest_impute_mad = mad_loss(data1, data2, data_m)
    randforest_impute_rmse = rmse_loss(data1, data2,data_m)
    return randforest_impute_mad, randforest_impute_rmse

#Decision Tree Imputation
def get_impute_tree(X_missing, ori_data, y_missing, data_m):
    temp = pd.DataFrame(X_missing.values, columns=X_missing.columns.astype(str).tolist())
    X_missing = temp
    y_missing = np.array(y_missing)
    X_missing_reg = X_missing.copy()
    X_df = X_missing_reg.isnull().sum()
    colname = X_df[~X_df.isin([0])].sort_values().index.values
    sortindex = []
    for i in colname:
        sortindex.append(X_missing_reg.columns.tolist().index(str(i)))

    for i in sortindex:
        df = X_missing_reg
        fillc = df.iloc[:, i]

        df = pd.concat([df.drop(df.columns[i], axis=1), pd.DataFrame(y_missing, columns=['y_missing'])], axis=1)

        df_0 = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0).fit_transform(df)

        Ytrain = fillc[fillc.notnull()]
        Ytest = fillc[fillc.isnull()]

        Xtrain = df_0[Ytrain.index, :]
        Xtest = df_0[Ytest.index, :]

        rfc = DecisionTreeRegressor()
        rfc = rfc.fit(Xtrain, Ytrain)
        Ypredict = rfc.predict(Xtest)

        X_missing_reg.loc[X_missing_reg.iloc[:, i].isnull(), X_missing_reg.columns[i]] = Ypredict
    # hp_data_re = hp_data.copy()
    # hp_data_re.iloc[:,start:end] = X_missing_reg
    # writer = pd.ExcelWriter('E:/FINAL/data_all_imputer.xlsx', engin='openpyxl')
    # book = load_workbook(writer.path)
    # writer.book = book
    # hp_data_re.to_excel(writer, sheet_name= 'tree Imputation', index=False)
    # writer.save()
    data1 = ori_data.values
    data2 = X_missing_reg.values
    # data1 = renormalization(data1, norm_parameters)
    # data2 = renormalization(data2, norm_parameters)
    tree_impute_mad = mad_loss(data1, data2, data_m)
    tree_impute_rmse = rmse_loss(data1, data2,data_m)
    return tree_impute_mad, tree_impute_rmse


#Random Forest for Final Imputation
def final_impute_randforest(X_missing, y_missing):
    temp = pd.DataFrame(X_missing.values, columns=X_missing.columns.astype(str).tolist())
    X_missing = temp
    y_missing = np.array(y_missing)
    # temp.iloc[:,:] = X_missing.iloc[:,:]
    X_missing_reg = X_missing.copy()
    X_df = X_missing_reg.isnull().sum()
    colname = X_df[~X_df.isin([0])].sort_values().index.values
    sortindex = []
    for i in colname:
        sortindex.append(X_missing_reg.columns.tolist().index(str(i)))

    for i in sortindex:
        df = X_missing_reg
        fillc = df.iloc[:, i]

        df = pd.concat([df.drop(df.columns[i], axis=1), pd.DataFrame(y_missing, columns=['y_missing'])], axis=1)
        df_0 = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0).fit_transform(df)

        Ytrain = fillc[fillc.notnull()]
        Ytest = fillc[fillc.isnull()]

        Xtrain = df_0[Ytrain.index, :]
        Xtest = df_0[Ytest.index, :]

        rfc = RandomForestRegressor()
        rfc = rfc.fit(Xtrain, Ytrain)
        Ypredict = rfc.predict(Xtest)

        X_missing_reg.loc[X_missing_reg.iloc[:, i].isnull(), X_missing_reg.columns[i]] = Ypredict
    # hp_data_re = hp_data.copy()
    # hp_data_re.iloc[:,start:end] = X_missing_reg
    # writer = pd.ExcelWriter('E:/FINAL/data_all_imputer.xlsx', engin='openpyxl')
    # book = load_workbook(writer.path)
    # writer.book = book
    # hp_data_re.to_excel(writer, sheet_name= 'Randforest Imputation', index=False)
    # writer.save()
    # data1 = ori_data.values
    # data2 = X_missing_reg.values
    # # data1 = renormalization(data1, norm_parameters)
    # # data2 = renormalization(data2, norm_parameters)
    # randforest_impute_mad = mad_loss(data1, data2, data_m)
    # randforest_impute_rmse = rmse_loss(data1, data2,data_m)
    return X_missing_reg.values
