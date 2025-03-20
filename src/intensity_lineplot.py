import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import skimage.io
import functools
from loguru import logger
from scipy.signal import savgol_filter
from scipy.interpolate import make_interp_spline
plt.rcParams.update({'font.size': 14})

logger.info('import OK')

input_folder = 'raw_data/'
plotting_folder = 'python_results/summary_calculations/'

if not os.path.exists(plotting_folder):
    os.mkdir(plotting_folder)

# ---------------- initialize file lists ----------------
# read in calculated pixel data
file_list = [filename for filename in os.listdir(input_folder) if '.xlsx' in filename]
pixels = {filename.replace('.xlsx', ''): pd.read_excel(f'{input_folder}{filename}', sheet_name=['Sheet1', 'Sheet2']) for filename in file_list}

for name, val in pixels.items():
    fig, ax = plt.subplots(figsize=(2.4, 2))
    max_val = val['Sheet2']['Gray_Value'].max()
    val['Sheet2']['gray_val_corrected'] = val['Sheet2']['Gray_Value']/max_val
    max_val = val['Sheet1']['Gray_Value'].max()
    val['Sheet1']['gray_val_corrected'] = val['Sheet1']['Gray_Value']/max_val
    mhat = savgol_filter(val['Sheet1']['gray_val_corrected'], 20, 4)
    ghat = savgol_filter(val['Sheet2']['gray_val_corrected'], 20, 4)
    plt.plot(val['Sheet1']['Distance_(microns)'], mhat, color='m')
    plt.plot(val['Sheet2']['Distance_(microns)'], ghat, color='g')
    plt.tight_layout()
    plt.savefig(f'{plotting_folder}{name}_intensitythumbnail.png', format='png', dpi=300)
