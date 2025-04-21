import os
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
import skimage.io
import functools
from skimage import measure, segmentation, morphology
from skimage.morphology import remove_small_objects
from scipy import stats
from scipy.stats import skew, skewtest
from loguru import logger
plt.rcParams.update({'font.size': 14})
sns.set_palette('Paired')

logger.info('import ok')

input_folder = 'python_results/initial_cleanup/'
mask_folder = 'python_results/napari_masking/'
output_folder = 'python_results/summary_calculations/'
proof_folder = 'python_results/proofs/'

if not os.path.exists(output_folder):
    os.mkdir(output_folder)

if not os.path.exists(proof_folder):
    os.mkdir(proof_folder)


def feature_extractor(mask, properties=False):

    if not properties:
        properties = ['area', 'eccentricity', 'label', 'major_axis_length', 'minor_axis_length', 'perimeter', 'coords']

    return pd.DataFrame(skimage.measure.regionprops_table(mask, properties=properties))


# ---------------- initialise file list ----------------
file_list = [filename for filename in os.listdir(
    input_folder) if 'npy' in filename]

images = {filename.replace('.npy', ''): np.load(
    f'{input_folder}{filename}') for filename in file_list}

# ---------------- important constants ----------------
std_threshold = 3.8

# --------------- process filtered masks for cytoplasm mask ---------------
filtered_masks = {masks.replace('_mask.npy', ''): np.load(
    f'{mask_folder}{masks}', allow_pickle=True) for masks in os.listdir(f'{mask_folder}') if '_mask.npy' in masks}

logger.info('removing nuclei from cell masks')
cytoplasm_masks = {}
for name, img in filtered_masks.items():
    name
    cell_mask = img[0, :, :]
    nuc_mask = img[1, :, :]
    # make binary masks
    cell_mask_binary = np.where(cell_mask, 1, 0)
    nuc_mask_binary = np.where(nuc_mask, 1, 0)
    single_cytoplasm_masks = []
    # need this elif in case images have no masks
    if len(np.unique(cell_mask).tolist()) > 1:
        for num in np.unique(cell_mask).tolist()[1:]:
            num
            # subtract whole nuclear mask per cell
            cytoplasm = np.where(cell_mask == num, cell_mask_binary, 0)
            cytoplasm_minus_nuc = np.where(cytoplasm == nuc_mask_binary, 0, cytoplasm)
            if np.count_nonzero(cytoplasm) != np.count_nonzero(cytoplasm_minus_nuc):
                # re-assign label
                cytoplasm_num = np.where(cytoplasm_minus_nuc, num, 0)
                single_cytoplasm_masks.append(cytoplasm_num)
            else:
                single_cytoplasm_masks.append(
                    np.zeros(np.shape(cell_mask)).astype(int))
    else:
        single_cytoplasm_masks.append(
        np.zeros(np.shape(cell_mask)).astype(int))
    # add cells together and update dict
    summary_array = sum(single_cytoplasm_masks)
    cytoplasm_masks[name] = summary_array
logger.info('nuclei removed')

# make dictionary from images and cytoplasm masks
image_mask_dict = {
    key: np.stack([images[key][0, :, :], images[key][3, :, :], cytoplasm_masks[key]])
    for key in images
}

# ---------------- remove saturated cells ----------------
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
        saturated_count = np.count_nonzero(cell == (2**16)-1) # assumes 16-bit image

        if (saturated_count/pixel_count) < 0.01:
            labels_filtered.append(label)

    cells_filtered = np.where(
        np.isin(image[2, :, :], labels_filtered), image[2, :, :], 0)

    # stack the filtered masks
    cells_filtered_stack = np.stack(
        (image[0, :, :], image[1, :, :], cells_filtered))
    not_saturated[name] = cells_filtered_stack
logger.info('saturated cells removed')

# ---------------- collect feature information ----------------
# now collect puncta and cell features info
logger.info('collecting cell and puncta feature info')
feature_information_list = []
for name, image in not_saturated.items():
    labels_filtered = []
    unique_val, counts = np.unique(image[2, :, :], return_counts=True)
    # find cell outlines for later plotting
    cell_binary_mask = np.where(image[2, :, :] !=0, 1, 0)
    contours = measure.find_contours(cell_binary_mask, 0.8)
    contour = [x for x in contours if len(x) >= 100]

    # loop to extract parameters from cells
    for num in unique_val[1:]:
        cell = np.where(image[2, :, :] == num, image[1, :, :], 0)
        cell_mean = np.mean(cell[cell != 0])
        cell_std = np.std(cell[cell != 0])
        # use std to threshold puncta 
        binary = (cell > (cell_std*std_threshold)).astype(int)
        # label thresholded puncta, remove small ones
        puncta_masks = measure.label(binary)
        puncta_masks = remove_small_objects(puncta_masks, 4**2)
        # extract puncta features
        cell_properties = feature_extractor(puncta_masks).add_prefix('puncta_')
        # collect stress granule channel info for later
        g3bp_cell = np.where(image[2, :, :] == num, image[0, :, :], 0)
        g3bp_cell_mean = np.mean(g3bp_cell[g3bp_cell != 0])

        # make lists for custom per-puncta measurements
        puncta_cv_list = []
        puncta_skew_list = []
        puncta_intensity_mean_list = []
        g3bp_intensity_mean_list = []
        for puncta_num in np.unique(puncta_masks)[1:]:
            puncta = np.where(puncta_masks == puncta_num, image[1,:,:], 0)
            g3bp = np.where(puncta!=0, image[0,:,:], 0)
            puncta = puncta[puncta!=0]
            g3bp = g3bp[g3bp!=0]
            puncta_cv = np.std(puncta) / np.mean(puncta)
            puncta_cv_list.append(puncta_cv)
            res = skewtest(puncta)
            puncta_skew = res.statistic
            puncta_skew_list.append(puncta_skew)
            puncta_intensity_mean_list.append(np.mean(puncta))
            g3bp_intensity_mean_list.append(np.mean(g3bp))
        # add measurements as columns
        cell_properties['puncta_cv'] = puncta_cv_list
        cell_properties['puncta_skew'] = puncta_skew_list
        cell_properties['puncta_intensity_mean'] = puncta_intensity_mean_list
        cell_properties['g3bp_intensity_mean'] = g3bp_intensity_mean_list
        
        # if a cell has no puncta, fill row with zeros
        if len(cell_properties) < 1:
            cell_properties.loc[len(cell_properties)] = 0

        # concatenate and add cell features
        properties = pd.concat([cell_properties])
        properties['image_name'] = name
        properties['cell_number'] = num
        properties['cell_size'] = np.size(cell[cell!=0])
        properties['cell_cv'] = cell_std / cell_mean
        res = skewtest(cell[cell!=0])
        properties['cell_skew'] = res.statistic
        properties['cell_rhm1_intensity_mean'] = cell_mean
        properties['cell_g3bp_intensity_mean'] = g3bp_cell_mean

        # add cell outlines for later proof plotting
        properties['cell_coords'] = [contour]*len(properties)

        # append properties to list
        feature_information_list.append(properties)
        
# concatenate feature info list into one df
feature_information = pd.concat(feature_information_list)
logger.info('completed feature collection')

# ---------------- wrangle feature information ----------------
# extract image metadata
feature_information['tag'] = feature_information['image_name'].str.split('-').str[0].str.split('_').str[-1]
feature_information['condition'] = feature_information['image_name'].str.split('_').str[2].str.split('-').str[0]
feature_information['rep'] = feature_information['image_name'].str.split('_').str[-1].str.split('-').str[0]

# add aspect ratio and puncta_circularity
feature_information['puncta_aspect_ratio'] = feature_information['puncta_minor_axis_length'] / feature_information['puncta_major_axis_length']
feature_information['puncta_circularity'] = (12.566*feature_information['puncta_area'])/(feature_information['puncta_perimeter']**2)

# add partitioning coeff for g3bp
feature_information['g3bp_partition_coeff'] = feature_information['g3bp_intensity_mean'] / feature_information['cell_g3bp_intensity_mean']
feature_information['rhm1_partition_coeff'] = feature_information['puncta_intensity_mean'] / feature_information['cell_rhm1_intensity_mean']

# remove outliers
puncta_features_of_interest = ['puncta_area', 'puncta_eccentricity', 'puncta_aspect_ratio', 'puncta_circularity', 'puncta_cv', 'puncta_skew', 'g3bp_partition_coeff', 'rhm1_partition_coeff', 'cell_cv', 'cell_skew']
for col in puncta_features_of_interest[:-1]:
    feature_information = feature_information[np.abs(stats.zscore(feature_information[col])) < 3]

# save data
feature_information.to_csv(f'{output_folder}puncta_features.csv')

# ---------------- make additional dataframes for averaging and normalization ----------------
# make additional df for avgs per replicate
feature_information_reps = []
for col in puncta_features_of_interest:
    reps_table = feature_information.groupby(['condition', 'tag', 'rep']).mean(numeric_only=True)[f'{col}']
    feature_information_reps.append(reps_table)
feature_information_reps_df = functools.reduce(lambda left, right: pd.merge(left, right, on=['condition', 'tag', 'rep'], how='outer'), feature_information_reps).reset_index()
feature_information_reps_df.to_csv(f'{output_folder}puncta_features_reps.csv')

# make additional dataframe normalized to cell intensity
normalized_features = feature_information.copy()
for col in puncta_features_of_interest:
    normalized_features[col] = normalized_features[col] / normalized_features['cell_rhm1_intensity_mean']
normalized_features.to_csv(f'{output_folder}puncta_features_normalized.csv')

# make additional df for avgs per replicate
normalized_features_reps = []
for col in puncta_features_of_interest:
    reps_table = normalized_features.groupby(['condition', 'tag', 'rep']).mean(numeric_only=True)[f'{col}']
    normalized_features_reps.append(reps_table)
normalized_features_reps_df = functools.reduce(lambda left, right: pd.merge(left, right, on=['condition', 'tag', 'rep'], how='outer'), normalized_features_reps).reset_index()
normalized_features_reps_df.to_csv(f'{output_folder}puncta_features_normalized_reps.csv')

logger.info('saved puncta feature dataframes')

# -------------- plot the proofs --------------
# generate a proof for each image including puncta segmentation overlaid
for name, image in image_mask_dict.items():
    unique_val, counts = np.unique(image[2, :, :], return_counts=True)

    # extract coords
    cell = np.where(image[2, :, :] != 0, image[1, :, :], 0)
    image_df = feature_information[(feature_information['image_name'] == name)]
    if len(image_df) > 0:
        cell_contour = image_df['cell_coords'].iloc[0]
        coord_list = np.array(image_df.puncta_coords)

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
        fig.savefig(f'{proof_folder}{name}_proof.png', bbox_inches='tight',pad_inches = 0.1, dpi = 300)
        plt.close()
        
logger.info('saved proofs')