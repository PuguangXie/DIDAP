# import tensorflow._api.v2.compat.v1 as tf
# tf.disable_v2_behavior()
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn import preprocessing
import warnings
warnings.filterwarnings("ignore")


#---------------- Normalization --------------------
def normalization(data, parameters=None):
    _, dim = data.shape
    norm_data = data.copy()

    if parameters is None:

        # MixMax normalization
        min_val = np.zeros(dim)
        max_val = np.zeros(dim)

        # For each dimension
        for i in range(dim):
            min_val[i] = np.nanmin(norm_data[:, i])
            norm_data[:, i] = norm_data[:, i] - np.nanmin(norm_data[:, i])
            max_val[i] = np.nanmax(norm_data[:, i])
            norm_data[:, i] = norm_data[:, i] / (np.nanmax(norm_data[:, i]) + 1e-6)

        # Return norm_parameters for renormalization
        norm_parameters = {'min_val': min_val, 'max_val': max_val}
    else:
        min_val = parameters['min_val']
        max_val = parameters['max_val']
        # For each dimension
        for i in range(dim):
            norm_data[:, i] = norm_data[:, i] - min_val[i]
            norm_data[:, i] = norm_data[:, i] / (max_val[i] + 1e-6)

        norm_parameters = parameters

    return norm_data, norm_parameters

def renormalization(norm_data, norm_parameters):
    min_val = norm_parameters['min_val']
    max_val = norm_parameters['max_val']

    _, dim = norm_data.shape
    renorm_data = norm_data.copy()

    for i in range(dim):
        renorm_data[:, i] = renorm_data[:, i] * (max_val[i] + 1e-6)
        renorm_data[:, i] = renorm_data[:, i] + min_val[i]

    return renorm_data

# temp_data = np.array(X_miss_hip_fracture.iloc[:,:])
# norm_data, norm_parameters = normalization(temp_data)
# X_miss_hip_fracture.iloc[:,:] = norm_data


def binary_sampler(p, rows, cols):
    unif_random_matrix = np.random.uniform(0., 1., size=[rows, cols])
    binary_random_matrix = 1 * (unif_random_matrix < p)
    return binary_random_matrix

def generative_missdata(data_x, miss_rate):

  # Parameters
  no, dim = data_x.shape

  # Introduce missing data
  data_m = binary_sampler(1-miss_rate, no, dim)
  miss_data_x = data_x.copy()
  miss_data_x.iloc[data_m == 0] = np.nan

  return data_x, miss_data_x, data_m

def rounding(imputed_data, data_x):
    rounded_data = imputed_data.copy()
    for i in range(data_x.shape[1]):
        non_missing = data_x[:, i][~np.isnan(data_x[:, i])]

        if np.all(np.mod(non_missing, 1) == 0):
            rounded_data[:, i] = np.round(rounded_data[:, i])

        elif len(np.unique(non_missing)) == 2:
            rounded_data[:, i] = np.round(rounded_data[:, i])
            rounded_data[:, i] = np.clip(rounded_data[:, i], 0, 1)
        else:
            pass

    return rounded_data


import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()
def xavier_init(shape):
    fan_in, fan_out = shape
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform(shape, minval=-limit, maxval=limit, dtype=tf.float32)


def uniform_sampler(low, high, rows, cols):
    return np.random.uniform(low, high, size=(rows, cols))


def sample_batch_index(total, batch_size):
    indices = np.arange(total)
    np.random.shuffle(indices)
    return indices[:batch_size]


def rmse_loss(ori_data, imputed_data, data_m):
    nominator = np.sum(((1 - data_m) * ori_data - (1 - data_m) * imputed_data) ** 2)
    denominator = np.sum(1 - data_m)

    rmse = np.sqrt(nominator / float(denominator))

    return rmse

def mad_loss(ori_data, imputed_data, data_m):
    nominator = np.sum(np.abs((1 - data_m) * ori_data - (1 - data_m) * imputed_data))
    denominator = np.sum(1 - data_m)

    mad = nominator / float(denominator)

    return mad


