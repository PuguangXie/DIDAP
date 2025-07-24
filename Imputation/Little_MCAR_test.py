import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr
import missingno as msno
import os
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
import rpy2.robjects as ro
from rpy2.robjects.packages import importr

def little_mcar_test(center_data):
    # ro.r("if(!require('remotes')) install.packages('remotes')")
    # ro.r("remotes::install_github('njtierney/naniar',force= T)")
    # pandas2ri.activate()
    naniar = importr('naniar')
    with localconverter(ro.default_converter + pandas2ri.converter):
        r_data = ro.conversion.py2rpy(center_data)
        result = naniar.mcar_test(r_data)
        chi_square = result['statistic'][0]
        # print(chi_square)
        df = result['df'][0]
        p_value = result['p.value'][0]

    if p_value > 0.05:
        conclusion = "Missing Completely At Random (MCAR)."
    else:
        conclusion = "NOT Missing Completely At Random."
    return {
        'chi_square': chi_square,
        'p_value': p_value,
        'df': df,
        'conclusion': conclusion
    }
