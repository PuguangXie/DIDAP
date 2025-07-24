import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from statsmodels.stats.stattools import medcouple
from scipy import stats
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from fancyimpute import BiScaler, KNN, SoftImpute, IterativeImputer as MICE
import statsmodels.api as sm
import seaborn as sns
from Missing import (missing, visualize_missingness, missing_heatmap)
from Little_MCAR_test import little_mcar_test
import warnings
warnings.filterwarnings("ignore")


################################# Data Loading ###########################
split=298
hp_data = pd.read_excel('../Data/raw_data.xlsx')    # missingness
# feature_names = hp_data.columns[:-1].tolist()
data = hp_data.iloc[:, :-1]
# print(data)
# print(hp_data.shape)
# y = data[:, -1]
centers = {
    'DXH': data[:split],
    'FAH': data[split:]
}
# center_output_dir = f"_results"
# os.makedirs(center_output_dir, exist_ok=True)

################################# Missingness Analysis ###########################
# Missingness Distribution Table
results = missing(centers)

# Missingness Distribution Barplot
visualize_missingness(centers)

for center_name, center_data in centers.items():
    print(f"\n========================= Processing data of {center_name} ===========================")

    # Missingness Heatmap
    missing_heatmap(center_name, center_data)

    # Little's MCAR test
    a = little_mcar_test(center_data)
    print(f"Chi-square statistic of Little's MCAR test is: {a['chi_square']}, degrees of freedom is: {a['df']}, and the p_value is: {a['p_value']}")
    print(f"So we conclude the {center_name} data is {a['conclusion']}")


