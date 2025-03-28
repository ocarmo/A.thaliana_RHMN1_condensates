import os
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
import skimage.io
import functools
import cv2
from skimage import measure, segmentation, morphology
from skimage.morphology import remove_small_objects
from skimage.feature import graycomatrix, graycoprops
from skimage.util import img_as_ubyte
from skimage.filters import threshold_otsu
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from scipy import ndimage
from scipy import stats
from scipy.stats import skewtest
from statannotations.Annotator import Annotator
from loguru import logger
plt.rcParams.update({'font.size': 14})
sns.set_palette('Paired')

logger.info('import ok')

input_folder = 'python_results/initial_cleanup/'
mask_folder = 'python_results/napari_masking/'
output_folder = 'python_results/summary_calculations/'
plotting_folder = 'python_results/plotting/'

if not os.path.exists(output_folder):
    os.mkdir(output_folder)

if not os.path.exists(plotting_folder):
    os.mkdir(plotting_folder)


def feature_extractor(mask, properties=False):

    if not properties:
        properties = ['area', 'eccentricity', 'label', 'major_axis_length', 'minor_axis_length', 'perimeter', 'coords']

    return pd.DataFrame(skimage.measure.regionprops_table(mask, properties=properties))


# ---------------- initialise file list ----------------
file_list = [filename for filename in os.listdir(
    input_folder) if 'npy' in filename]

images = {filename.replace('.npy', ''): np.load(
    f'{input_folder}{filename}') for filename in file_list}

masks = np.load(f'{mask_folder}cytoplasm_masks.npy',
                allow_pickle=True).item()

# make dictionary from images and masks array
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
        cell = np.where(image[2, :, :] == label, image[1, :, :], 0)
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
logger.info('collecting cell and puncta feature info')
feature_information_list = []
for name, image in not_saturated.items():
    # logger.info(f'Processing {name}')
    labels_filtered = []
    unique_val, counts = np.unique(image[2, :, :], return_counts=True)
    # find cell outlines for later plotting
    cell_binary_mask = np.where(image[2, :, :] !=0, 1, 0)
    contours = measure.find_contours(cell_binary_mask, 0.8)
    contour = [x for x in contours if len(x) >= 100]
    # loop to extract params from cells
    for num in unique_val[1:]:
        num
        cell = np.where(image[2, :, :] == num, image[1, :, :], 0)
        cell_mean = np.mean(cell[cell != 0])
        cell_std = np.std(cell[cell != 0])
        binary = (cell > (cell_std*3.8)).astype(int)
        puncta_masks = measure.label(binary)
        puncta_masks = remove_small_objects(puncta_masks, 4**2)
        cell_properties = feature_extractor(puncta_masks).add_prefix('granule_')
        g3bp_cell = np.where(image[2, :, :] == num, image[0, :, :], 0)
        g3bp_cell_mean = np.mean(g3bp_cell[g3bp_cell != 0])

        # make list for cov and skew, add as columns to properties
        granule_cov_list = []
        granule_skew_list = []
        granule_intensity_list = []
        g3bp_intensity_list = []
        for granule_num in np.unique(puncta_masks)[1:]:
            granule_num
            granule = np.where(puncta_masks == granule_num, image[1,:,:], 0)
            g3bp = np.where(granule!=0, image[0,:,:], 0)
            granule = granule[granule!=0]
            g3bp = g3bp[g3bp!=0]
            granule_cov = np.std(granule) / np.mean(granule)
            granule_cov_list.append(granule_cov)
            res = skewtest(granule)
            granule_skew = res.statistic
            granule_skew_list.append(granule_skew)
            granule_intensity_list.append(np.mean(granule))
            g3bp_intensity_list.append(np.mean(g3bp))
        cell_properties['granule_cov'] = granule_cov_list
        cell_properties['granule_skew'] = granule_skew_list
        cell_properties['granule_intensity'] = granule_intensity_list
        cell_properties['g3bp_intensity'] = g3bp_intensity_list
        
        if len(cell_properties) < 1:
            cell_properties.loc[len(cell_properties)] = 0

        properties = pd.concat([cell_properties])
        properties['image_name'] = name
        properties['cell_number'] = num
        properties['cell_size'] = np.size(cell[cell!=0])
        properties['cell_cov'] = cell_std / cell_mean
        res = skewtest(cell[cell!=0])
        properties['cell_skew'] = res.statistic
        properties['cell_rhm1_intensity_mean'] = cell_mean
        properties['cell_g3bp_intensity_mean'] = g3bp_cell_mean

        # add cell outlines to coords
        properties['cell_coords'] = [contour]*len(properties)

        feature_information_list.append(properties)
        
feature_information = pd.concat(feature_information_list)
logger.info('completed feature collection')

# extract image metadata
feature_information['tag'] = feature_information['image_name'].str.split('-').str[0].str.split('_').str[-1]
feature_information['condition'] = feature_information['image_name'].str.split('_').str[2].str.split('-').str[0]
feature_information['rep'] = feature_information['image_name'].str.split('_').str[-1].str.split('-').str[0]

# add aspect ratio and granule_circularity
feature_information['granule_aspect_ratio'] = feature_information['granule_minor_axis_length'] / feature_information['granule_major_axis_length']
feature_information['granule_circularity'] = (12.566*feature_information['granule_area'])/(feature_information['granule_perimeter']**2)

# add partitioning coeff for g3bp
feature_information['g3bp_partition_coeff'] = feature_information['g3bp_intensity'] / feature_information['cell_g3bp_intensity_mean']

# remove outliers
granule_features_of_interest = ['granule_area', 'granule_eccentricity', 'granule_aspect_ratio', 'granule_circularity', 'granule_cov', 'granule_skew', 'g3bp_partition_coeff', 'cell_cov', 'cell_skew']
for col in granule_features_of_interest[:-1]:
    feature_information = feature_information[np.abs(stats.zscore(feature_information[col])) < 3]

# save data for plotting coords
feature_information.to_csv(f'{output_folder}puncta_detection_feature_info.csv')

# make additional df for avgs per replicate
granule_summary_reps = []
for col in granule_features_of_interest:
    reps_table = feature_information.groupby(['condition', 'tag', 'rep']).mean(numeric_only=True)[f'{col}']
    granule_summary_reps.append(reps_table)
granule_summary_reps_df = functools.reduce(lambda left, right: pd.merge(left, right, on=['condition', 'tag', 'rep'], how='outer'), granule_summary_reps).reset_index()

# -------------- calculate feature information per cell --------------
# grab major and granule_minor_axis_length for punctas
minor_axis = feature_information.groupby(
    ['image_name', 'cell_number'])['granule_minor_axis_length'].mean()
major_axis = feature_information.groupby(
    ['image_name', 'cell_number'])['granule_major_axis_length'].mean()

# calculate average size of punctas per cell
puncta_avg_area = feature_information.groupby(
    ['image_name', 'cell_number'])['granule_area'].mean().reset_index()

# calculate proportion of area in punctas
cell_size = feature_information.groupby(
    ['image_name', 'cell_number'])['cell_size'].mean()
puncta_area = feature_information.groupby(
    ['image_name', 'cell_number'])['granule_area'].sum()
puncta_proportion = ((puncta_area / cell_size) *
                   100).reset_index().rename(columns={0: 'proportion_puncta_area'})

# calculate number of 'punctas' per cell
puncta_count = feature_information.groupby(
    ['image_name', 'cell_number'])['granule_area'].count()

# calculate average size of punctas per cell
avg_eccentricity = feature_information.groupby(
    ['image_name', 'cell_number'])['granule_eccentricity'].mean().reset_index()

# grab cell intensity mean 
granule_cov_mean = feature_information.groupby(
    ['image_name', 'cell_number'])['granule_cov'].mean()

# grab cell intensity mean 
granule_skew_mean = feature_information.groupby(
    ['image_name', 'cell_number'])['granule_skew'].mean()

# grab cell intensity mean 
cell_rhm1_intensity_mean = feature_information.groupby(
    ['image_name', 'cell_number'])['cell_rhm1_intensity_mean'].mean()

# grab cell g3bp1 partitioning coefficient 
g3bp_partition_coeff = feature_information.groupby(
    ['image_name', 'cell_number'])['g3bp_partition_coeff'].mean()

# grab cell cell rhm1 cov 
cell_cov = feature_information.groupby(
    ['image_name', 'cell_number'])['cell_cov'].mean()

# grab cell cell rhm1 skew 
cell_skew = feature_information.groupby(
    ['image_name', 'cell_number'])['cell_skew'].mean()

# summarise, save to csv
summary = functools.reduce(lambda left, right: pd.merge(left, right, on=['image_name', 'cell_number'], how='outer'), [cell_size.reset_index(), puncta_avg_area, puncta_proportion, puncta_count.reset_index(), minor_axis, major_axis, avg_eccentricity, granule_cov_mean, granule_skew_mean, g3bp_partition_coeff, cell_cov, cell_skew, cell_rhm1_intensity_mean])
summary.columns = ['image_name', 'cell_number',  'cell_size', 'mean_puncta_area', 'puncta_area_proportion', 'puncta_count', 'puncta_mean_minor_axis', 'puncta_mean_major_axis', 'avg_eccentricity', 'granule_cov_mean', 'granule_skew_mean', 'g3bp_partition_coeff', 'cell_cov', 'cell_skew', 'cell_rhm1_intensity_mean']

# extract image metadata
summary['tag'] = summary['image_name'].str.split('-').str[0].str.split('_').str[-1]
summary['condition'] = summary['image_name'].str.split('_').str[2].str.split('-').str[0]
summary['rep'] = summary['image_name'].str.split('_').str[-1].str.split('-').str[0]

# features of interest
cell_features_of_interest = summary.columns.tolist()[3:-3]

# remove outliers
for col in cell_features_of_interest:
    summary = summary[np.abs(stats.zscore(summary[col])) < 3]

# save summary
summary.to_csv(f'{output_folder}puncta_detection_summary.csv')

# average data by biological replicate
cell_summary_reps = []
for col in cell_features_of_interest:
    reps_table = summary.groupby(['condition', 'tag', 'rep']).mean(numeric_only=True)[f'{col}']
    cell_summary_reps.append(reps_table)
cell_summary_reps_df = functools.reduce(lambda left, right: pd.merge(left, right, on=['condition', 'tag', 'rep'], how='outer'), cell_summary_reps).reset_index()

# normalize to mean cell intensity
normalized_summary = summary.copy()
for col in cell_features_of_interest:
    normalized_summary[col] = normalized_summary[col] / normalized_summary['cell_rhm1_intensity_mean']

# average normalized data by biological replicate
norm_cell_summary_reps = []
for col in cell_features_of_interest:
    reps_table = normalized_summary.groupby(['condition', 'tag', 'rep']).mean(numeric_only=True)[f'{col}']
    norm_cell_summary_reps.append(reps_table)
norm_cell_summary_reps_df = functools.reduce(lambda left, right: pd.merge(left, right, on=['condition', 'tag', 'rep'], how='outer'), norm_cell_summary_reps).reset_index()

# -------------- plotting data --------------
# suptitle, features_of_interest, raw_data, averaged_data, save_name
plotting_list = [
    ['per granule, not normalized', granule_features_of_interest, feature_information, granule_summary_reps_df,'puncta-features_pergranule_raw'],
    ['per granule, normalized to cell intensity', granule_features_of_interest, feature_information_norm, granule_summary_reps_norm_df, 'puncta-features_pergranule_normalized'],
    ['per cell, not normalized to cytoplasm intensity', cell_features_of_interest, summary, cell_summary_reps_df, 'puncta-features_percell_raw'],
    ['per cell, normalized to cytoplasm intensity', cell_features_of_interest, normalized_summary, norm_cell_summary_reps_df, 'puncta-features_percell_normalized']
]

# plot by pairing FLAG and GFP tags, including stats
pairs = [(('PBS', 'GFP'), ('PBS', 'FLAG')),
        (('NaAsO2', 'GFP'), ('NaAsO2', 'FLAG')),
        (('HS', 'GFP'), ('HS', 'FLAG'))]
order = ['PBS', 'NaAsO2', 'HS']
x = 'condition'
y = 'tag'
for sublist in plotting_list:
    plt.figure(figsize=(15, 15))
    plt.subplots_adjust(hspace=0.5)
    plt.suptitle(f'calculated parameters - {sublist[0]}', fontsize=18, y=0.99)
    for n, parameter in enumerate(sublist[1]):

        # add a new subplot iteratively
        ax = plt.subplot(4, 3, n + 1)

        # filter df and plot ticker on the new subplot axis
        sns.stripplot(data=sublist[2], x=x, y=parameter, dodge='True', 
                        edgecolor='white', linewidth=1, size=8, alpha=0.4, hue=y, order=order, ax=ax)
    
        # store legends info
        handles, labels = ax.get_legend_handles_labels()

        # continue plotting
        sns.stripplot(data=sublist[3], x=x, y=parameter, dodge='True', edgecolor='k', linewidth=1, size=8, hue=y, order=order, ax=ax)
        sns.boxplot(data=sublist[3], x=x, y=parameter,
                    palette=['.9'], hue=y, order=order, ax=ax)

        # remove all legends
        ax.legend().remove()

        # statannot stats
        annotator = Annotator(ax, pairs, data=sublist[3], x=x, y=parameter, hue=y, order=order)
        annotator.configure(test='Mann-Whitney', verbose=2)
        annotator.apply_test()
        annotator.annotate()

        # formatting
        sns.despine()
        ax.set_xticklabels(['PBS', r'NaAsO$_{2}$', 'HS'])

    plt.tight_layout()
    plt.legend(handles, labels, bbox_to_anchor=(1.1, 1), title='RHM1 tag')
    plt.savefig(f'{output_folder}tag-paired_{sublist[4]}.png', bbox_inches='tight',pad_inches = 0.1, dpi = 300)


# plot by pairing conditions
palette = ['#A6CEE3', '#1F78B4', '#F5CB5C']
order = ['PBS', 'NaAsO2', 'HS']
x = 'tag'
y = 'condition'
for sublist in plotting_list:
    sublist
    plt.figure(figsize=(15, 15))
    plt.subplots_adjust(hspace=0.5)
    plt.suptitle(f'calculated parameters - {sublist[0]}', fontsize=18, y=0.99)
    for n, parameter in enumerate(sublist[1]):

        # add a new subplot iteratively
        ax = plt.subplot(4, 3, n + 1)

        # filter df and plot ticker on the new subplot axis
        sns.stripplot(data=sublist[3], x=x, y=parameter, dodge='True', edgecolor='k', linewidth=1, size=8, hue=y, palette=palette, hue_order=order, zorder=2, ax=ax)
    
        # store legends info
        handles, labels = ax.get_legend_handles_labels()

        # continue plotting
        sns.stripplot(data=sublist[2], x=x, y=parameter, dodge='True', 
                        edgecolor='white', linewidth=1, size=8, alpha=0.4, hue=y, palette=palette, hue_order=order, zorder=1, ax=ax)
        sns.boxplot(data=sublist[3], x=x, y=parameter,
                    palette=['.9'], hue=y, hue_order=order, zorder=0, ax=ax)

        # remove all legends
        ax.legend().remove()

        # formatting
        sns.despine()
        ax.set_xticklabels(['FLAG-RHM1', 'GFP-RHM1'])
        ax.set_xlabel('')

    plt.tight_layout()
    plt.legend(handles, labels, bbox_to_anchor=(1.1, 1), title='Condition')
    ax.legend(labels=['PBS', r'NaAsO$_{2}$', 'HS'])
    plt.savefig(f'{output_folder}condition-paired_{sublist[4]}.png', bbox_inches='tight',pad_inches = 0.1, dpi = 300)

# -------------- plotting proofs --------------
# plot proofs
for name, image in image_mask_dict.items():
    name
    unique_val, counts = np.unique(image[2, :, :], return_counts=True)

    # extract coords
    cell = np.where(image[2, :, :] != 0, image[1, :, :], 0)
    image_df = feature_information[(feature_information['image_name'] == name)]
    if len(image_df) > 0:
        cell_contour = image_df['cell_coords'].iloc[0]
        coord_list = np.array(image_df.granule_coords)

        # plot
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(image[1,:,:], cmap=plt.cm.gray_r)
        ax1.imshow(image[0,:,:], cmap=plt.cm.Blues, alpha=0.60)
        ax2.imshow(cell, cmap=plt.cm.gray_r)
        for cell_line in cell_contour:
            ax2.plot(cell_line[:, 1], cell_line[:, 0], linewidth=0.5, c='k')
        if len(coord_list) > 1:
            for puncta in coord_list:
                if isinstance(puncta, np.ndarray):
                    ax2.plot(puncta[:, 1], puncta[:, 0], linewidth=0.5)
        for ax in fig.get_axes():
            ax.label_outer()

        # create scale bar and labels
        scalebar = ScaleBar(0.0779907, 'um', location = 'lower right', pad = 0.3, sep = 2, box_alpha = 0, color='gray', length_fraction=0.3)
        ax1.add_artist(scalebar)
        ax1.text(50, 2000, 'RHM1', color='gray')
        ax1.text(50, 1800, 'G3BP', color='steelblue')

        # title and save
        fig.suptitle(name, y=0.78)
        fig.tight_layout()
        fig.savefig(f'{plotting_folder}{name}_proof.png', bbox_inches='tight',pad_inches = 0.1, dpi = 300)
        plt.close()
