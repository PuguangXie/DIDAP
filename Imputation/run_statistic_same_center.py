import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from openpyxl import load_workbook
from scipy import stats

hp_data = pd.read_excel('D:/GAIN-master/data/raw_data.xlsx')
##between dead and live, same center
data = np.array(hp_data.iloc[298:, :])

dead_group = data[np.where(data[:, -1] == 1)]
live_group = data[np.where(data[:, -1] == 0)]

# continuece_index = [1, 4, 5, 16, 17, 18, 19, 20, 21]
discrete_index = [0, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

index = discrete_index

dead_data = dead_group[:, index]
live_data = live_group[:, index]

data = data[:, index]

num = 0
for i in (hp_data.columns[index]):
    print(i)
    # miss_rate = np.sum(pd.isnull(data[:, index])) / data.shape[0]
    # print('missing_rate：', miss_rate)
    # dead_column = (dead_data[:, index])[pd.notnull(dead_data[:, index])]
    # print('dead-正态性：', stats.normaltest(dead_column)[1])
    # live_column = (live_data[:, index])[pd.notnull(live_data[:, index])]
    # print('live-正态性：', stats.normaltest(live_column)[1])
    #
    # print('dead_mean：', np.mean(dead_column), '------', 'dead_std：', np.std(dead_column))
    # print('live_mean：', np.mean(live_column), '------', 'live_std：', np.std(live_column))
    #
    # res = stats.mannwhitneyu(dead_column, live_column, alternative='two-sided')
    # print(res)

    ##discrete index
    dead_column = dead_data[:, num]
    live_column = live_data[:, num]
    a = np.sum(dead_column)
    b = dead_column.shape[0] - a
    c = np.sum(live_column)
    d = live_column.shape[0] - c
    frame = np.array([[a, c], [b, d]])
    res = stats.chi2_contingency(frame)
    T = res[3]
    if np.min(T) < 5:
        res = stats.chi2_contingency(frame, True)
    num = num + 1
    print('a:', a, 'b:', b)
    print('c:', c, 'd:', d)
    print(res)



