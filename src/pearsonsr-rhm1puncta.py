import numpy as np
from scipy import stats
import os
import numpy as np
import pandas as pd
from functools import reduce
from loguru import logger
import seaborn as sns
import skimage.io
from skimage import measure
from skimage.morphology import remove_small_objects
from scipy.stats import mannwhitneyu, normaltest
from statannotations.Annotator import Annotator
import functools
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})

logger.info('Import OK')

# define location parameters
image_folder = 'python_results/initial_cleanup/'
mask_folder = 'python_results/napari_masking/'
output_folder = 'python_results/summary_calculations/'
plotting_folder = 'python_results/plotting/'

if not os.path.exists(output_folder):
    os.mkdir(output_folder)
if not os.path.exists(plotting_folder):
    os.mkdir(plotting_folder)

# ---------------- initialise file list ----------------
file_list = [filename for filename in os.listdir(
    image_folder) if 'npy' in filename]

images = {filename.replace('.npy', ''): np.load(
    f'{image_folder}{filename}') for filename in file_list}

masks = np.load(f'{mask_folder}cytoplasm_masks.npy', allow_pickle=True).item()

image_mask_dict = {
    key: np.stack([images[key][0, :, :], images[key][3, :, :], masks[key]])
    for key in images
}

# ---------------- collect feature information ----------------
# remove saturated cells in case some were added during manual validation
logger.info('removing saturated cells')
not_saturated = {}
for name, image in image_mask_dict.items():
    labels_filtered = []
    unique_val, counts = np.unique(image[2, :, :], return_counts=True)

    # loop to remove saturated cells (>1% px values > 60000)
    for label in unique_val[1:]:
        pixel_count = np.count_nonzero(image[2, :, :] == label)
        cell = np.where(image[2, :, :] == label, image[0, :, :], 0)
        saturated_count = np.count_nonzero(cell == 65535)

        if (saturated_count/pixel_count) < 0.01:
            labels_filtered.append(label)

    cells_filtered = np.where(
        np.isin(image[2, :, :], labels_filtered), image[2, :, :], 0)

    # stack the filtered masks
    cells_filtered_stack = np.stack(
        (image[0, :, :], image[1, :, :], cells_filtered))
    not_saturated[name] = cells_filtered_stack

# now collect puncta and cell features info
logger.info('collecting Pearsons correlation coefficient')
pearsonsrs = {}
pearsonsrs_df = pd.DataFrame(columns=['cell', 'pearsonsr'])
for name, image in not_saturated.items():
    name
    unique_val, counts = np.unique(image[2, :, :], return_counts=True)
    # find cell outlines for later plotting
    cell_binary_mask = np.where(image[2, :, :] !=0, 1, 0)
    contours = measure.find_contours(cell_binary_mask, 0.8)
    contour = [x for x in contours if len(x) >= 100]
    # loop to extract params from cells
    for num in unique_val[1:]:
        num
        cell = np.where(image[2, :, :] == num, image[1, :, :], 0)
        cell_std = np.std(cell[cell != 0])
        cell_mean = np.mean(cell[cell != 0])
        binary = (cell > (cell_std*3.8)).astype(int)
        puncta_masks = measure.label(binary)
        puncta_masks = remove_small_objects(puncta_masks, 4**2)
        rhm1_probe = np.where(puncta_masks != 0, cell, 0) 
        stress_probe = np.where(image[2, :, :] == num, image[0, :, :], 0)
        res = stats.pearsonr(stress_probe.flatten(), rhm1_probe.flatten())[0]
        pearsonsrs[(name, num)] = res   

coeff = pd.DataFrame(list(pearsonsrs.items()), columns=[
             'name', 'pearsonsr'])
coeff[['image_name', 'cell_number']] = coeff['name'].apply(pd.Series)
coeff.drop('name', axis=1, inplace=True)

# extract image metadata
coeff['tag'] = coeff['image_name'].str.split('-').str[0].str.split('_').str[-1]
coeff['condition'] = coeff['image_name'].str.split('_').str[2].str.split('-').str[0]
coeff['rep'] = coeff['image_name'].str.split('_').str[-1].str.split('-').str[0]

# ---------------- calculate mean and std ----------------
# determine mean per cell per channel
mean_r = coeff.groupby(['tag', 'condition', 'rep'])['pearsonsr'].mean()
std_r = coeff.groupby(['tag', 'condition', 'rep'])['pearsonsr'].std()

summary = functools.reduce(lambda left, right: pd.merge(
    left, right, on=['tag', 'condition', 'rep'], how='outer'), [mean_r, std_r]).reset_index()
summary.columns = ['tag', 'condition', 'rep', 'mean_r', 'std_r']

# # n values
# coeff[coeff['construct']=='48'].groupby(['condition','construct']).count()['pearsonsr']
# # condition  construct
# # nNaAsO2    48            96
# # yNaAsO2    48           122
# # Name: pearsonsr, dtype: int64

# ---------------- plotting ----------------
sns.set_palette('Paired')
pairs = [(('PBS', 'GFP'), ('PBS', 'FLAG')),
        (('NaAsO2', 'GFP'), ('NaAsO2', 'FLAG')),
        (('HS', 'GFP'), ('HS', 'FLAG'))]
order = ['PBS', 'NaAsO2', 'HS']
x = 'condition'
y = 'tag'

fig, ax = plt.subplots(figsize=(5, 4))
sns.stripplot(data=coeff, x=x, y='pearsonsr', dodge='True', 
                edgecolor='white', linewidth=1, size=8, alpha=0.4, hue=y, order=order, ax=ax)

# store legends info
handles, labels = ax.get_legend_handles_labels()

# continue plotting
sns.stripplot(data=summary, x=x, y='mean_r', dodge='True', edgecolor='k', linewidth=1, size=8, hue=y, order=order, ax=ax)
sns.boxplot(data=summary, x=x, y='mean_r',
            palette=['.9'], hue=y, order=order, ax=ax)

# remove all legends
ax.legend().remove()

# statannot stats
annotator = Annotator(ax, pairs, data=summary, x=x, y='mean_r', hue=y, order=order)
annotator.configure(test='Mann-Whitney', verbose=2)
annotator.apply_test()
annotator.annotate()

# formatting
sns.despine()
# ax.set_xlabel('')
ax.set_xticklabels(['PBS', r'NaAsO$_{2}$', 'HS'])
plt.ylabel('Pearsons correlation coefficient')

plt.tight_layout()
plt.legend(handles, labels, bbox_to_anchor=(1.1, 1), title='RHM1 tag')
plt.savefig(f'{output_folder}pearsons-coeff.png', bbox_inches='tight',pad_inches = 0.1, dpi = 300)
