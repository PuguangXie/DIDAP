import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from openpyxl import load_workbook
from scipy import stats


hp_data = pd.read_excel('./raw_data.xlsx')
##different center

DP_data = np.array(hp_data.iloc[:298, :])
CY_data = np.array(hp_data.iloc[298:, :])


continuece_index = [1, 4, 5, 16, 17, 18, 19, 20, 21]
discrete_index = [0, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

index = discrete_index

DP_data = DP_data[:, index]
CY_data = CY_data[:, index]

num = 0

for i in (hp_data.columns[index]):
    print(i)

    # DP_column = (DP_data[:, num])[pd.notnull(DP_data[:, num])]
    # print('DP-正态性：', stats.normaltest(DP_column)[1])
    # CY_column = (CY_data[:, num])[pd.notnull(CY_data[:, num])]
    # print('CY-正态性：', stats.normaltest(CY_column)[1])
    #
    # print('DP_mean：', np.mean(DP_column), '------', 'DP_std：', np.std(DP_column))
    # print('CY_mean：', np.mean(CY_column), '------', 'CY_std：', np.std(CY_column))
    #
    # res = stats.mannwhitneyu(DP_column, CY_column, alternative='two-sided')
    # num = num + 1
    # print(res)

    ##discrete index
    DP_column = DP_data[:, num]
    CY_column = CY_data[:, num]
    a = np.sum(DP_column)
    b = DP_column.shape[0] - a
    c = np.sum(CY_column)
    d = CY_column.shape[0] - c
    frame = np.array([[a, c], [b, d]])
    res = stats.chi2_contingency(frame)
    T = res[3]
    if np.min(T) < 5:
        res = stats.chi2_contingency(frame, True)
    num = num + 1
    print('a:', a, 'b:', b)
    print('c:', c, 'd:', d)
    print(res)



