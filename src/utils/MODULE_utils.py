import os
import numpy as np
from scipy.optimize import curve_fit
import pandas as pd

def clean_up_folder(mother_folder):
    for root, dirs, files in os.walk(mother_folder):
        for name in files:
            if 'checkpoint' in name:
                os.remove(f'{root}/{name}')
    
    for root, dirs, files in os.walk(mother_folder):
        for name in dirs:
            if 'checkpoint' in name:
                os.rmdir(f'{root}/{name}')

def check_path(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def scaled_expo(x,C,lamb):
    return C*np.exp(-lamb*x)

def get_lambda(list_in): # [p0, p1, p2, p3, p4]
    x_vals = np.array([0.3, 0.9, 1.5, 2.1, 2.7])    
    try:
        params, _ = curve_fit(scaled_expo, x_vals, list_in)
        return params[1]
    except:
        return 99.9

def cal_SPASL_score(prob_in):
    assert len(prob_in) == 5, 'Invalid length, should be 5'
    cur_sum = 0
    for i in range(4):
        cur_sum += prob_in[i]*prob_in[i+1]
    cur_sum += prob_in[-1]*prob_in[0]
    return round(cur_sum/500, 2)

def cross_search_for_top_n_images(IW_table_csv_file, n, TYPE):
    df0 = pd.read_csv(IW_table_csv_file)
    assert TYPE == 'I' or TYPE == 'IV', 'Valid TYPE in this function is I or IV'
    if TYPE == 'I':
        cond1 = 'R1_Count'
    elif TYPE == 'IV':
        cond1 = 'Wrong_Count'
    df1 = df0.sort_values(by = [cond1], ascending=False)
    df2 = df0.sort_values(by = ['lamb_Q3'], ascending=False)
    
    for i in range(n, len(df0), 100):
        df1_part = df1.reset_index(drop=True).loc[:i]
        df2_part = df2.reset_index(drop=True).loc[:i]
        in_df = pd.merge(df1_part, df2_part, how='inner')
        if len(in_df) > n:
            return in_df.loc[:n]

