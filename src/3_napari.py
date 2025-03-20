"""Quality control: use napari to validate cellpose-generated masks
"""

from scipy import ndimage
from loguru import logger
import napari
import numpy as np
import matplotlib.pyplot as plt
import os
import collections
import skimage.io
from skimage.measure import label
from skimage.segmentation import clear_border

image_folder = 'python_results/initial_cleanup/'
mask_folder = 'python_results/cellpose_masking/'
output_folder = 'python_results/napari_masking/'

if not os.path.exists(output_folder):
    os.mkdir(output_folder)


def filter_masks(before_image, image_name, mask):
    """Quality control of cellpose-generated masks
    - Select the cell layer and using the fill tool set to 0, remove all unwanted cells.
    - Finally, using the brush tool add or adjust any masks within the appropriate layer.

    Args:
        before_image (ndarray): self explanatory
        image_name (str): self explanatory
        mask (ndarray): self explanatory

    Returns:
        ndarray: stacked masks
    """
    cells = mask[0, :, :].copy()
    nuclei = mask[1, :, :].copy()
    
    viewer = napari.Viewer()

    # create the viewer and add the image
    viewer = napari.view_image(before_image, name='before_image')
    # add the labels
    viewer.add_labels(cells, name='cells')
    viewer.add_labels(nuclei, name='nuclei')

    napari.run()

    np.save(f'{output_folder}{image_name}_mask.npy',
            np.stack([cells, nuclei]))
    logger.info(
        f'Processed {image_name}. Mask saved to {output_folder}{image_name}')

    return np.stack([cells, nuclei])


def stack_channels(name, masks_filtered, cells_filtered_stack):
    masks_filtered[name] = cells_filtered_stack

# ---------------- initialise file list ----------------
# read in numpy masks
cell_masks = np.load(f'{mask_folder}cellpose_cellmasks.npy')
nuc_masks = np.load(f'{mask_folder}cellpose_nucmasks.npy')

# clean filenames
file_list = [filename for filename in os.listdir(
    image_folder) if 'npy' in filename]

# 0 = before; 1 = after; 2 = h342; 3 = POL
images = {filename.replace('.npy', ''): np.load(
    f'{image_folder}{filename}') for filename in file_list}

raw_stacks = {
    image_name: np.stack([cell_masks[x, :, :], nuc_masks[x, :, :]])
    for x, image_name in (enumerate(images.keys()))}

# ---------------- erode cell stacks ----------------
logger.info('start eroding cell masks')
raw_masks_eroded = {}
structure_element = np.ones((4, 4)).astype(int) # for erosion
for name, img in raw_stacks.items():
    name
    nuc_mask = img[1,:,:]
    single_cells = []
    # make sure image has cell masks, else array of 0s
    if len(np.unique(img[0, :, :]).tolist()[1:]) > 1:
        # for mask in image
        for num in np.unique(img[0, :, :]).tolist()[1:]:
            num
            # make cell binary
            cell = np.where(img[0,:,:] == num, 1, 0)
            # make 'footprint' and erode mask
            cell_eroded = ndimage.binary_erosion(
                cell, structure_element).astype(cell.dtype)
            # reassign binary to 0 and num
            mask_eroded = np.where(cell_eroded == 1, num, 0)
            # if cell large enough (but not 1024**2)
            if np.unique(mask_eroded, return_counts=True)[-1][-1] > 4000 and np.unique(mask_eroded, return_counts=True)[-1][-1] < 1048576:
                single_cells.append(mask_eroded)
            else:
                single_cells.append(np.zeros(np.shape(mask_eroded)).astype(int))
    else:
        single_cells.append(np.zeros(np.shape(nuc_mask)).astype(int))
    # add cells together and update dict
    sum_array = sum(single_cells)
    mask_stack = np.stack([sum_array, nuc_mask])
    raw_masks_eroded[name] = mask_stack
logger.info('completed cell mask eroding')

# make new dictionary to check for saturation
image_names = images.keys()
image_values = zip(images.values(), raw_masks_eroded.values())
saturation_check = dict(zip(image_names, image_values))

# ---------------- filtering saturated cells and/or near border ----------------
masks_filtered = {}
logger.info('removing saturated and/or border cells')
for name, image in saturation_check.items():
    labels_filtered = []
    # image order: [0] before, after, h342, pol
    # image order: [1] cell_mask, nuc_mask
    # find cells with few pixels and remove
    unique_val, counts = np.unique(image[1][0, :, :], return_counts=True)

    # loop to remove 'before' saturated cells (>5% px values > 60000)
    for label in unique_val[1:]:
        label
        pixel_count = np.count_nonzero(image[1][0, :, :] == label)
        cell = np.where(image[1][0, :, :] == label, image[0][3, :, :], 0)
        saturated_count = np.count_nonzero(cell > 60000)

        if (saturated_count/pixel_count) < 0.05:
            labels_filtered.append(label)

    cells_filtered = np.where(
        np.isin(image[1][0, :, :], labels_filtered), image[1][0, :, :], 0)
    
    # remove cells near border
    cells_filtered = clear_border(cells_filtered, buffer_size=10)
    
    # keep intracellular nuclei
    intra_nuclei = np.where(
        cells_filtered >= 1, image[1][1, :, :], 0)
    
    # filter out small nuclei
    nuc_unique_val, nuc_counts = np.unique(
        intra_nuclei, return_counts=True)
    for nuc_label in nuc_unique_val[1:]:
        nuc_test = np.where(intra_nuclei == nuc_label, nuc_label, 0)
        if np.unique(nuc_test, return_counts=True)[-1][-1] < 8000:
            intra_nuclei = np.where(intra_nuclei == nuc_label, 0, intra_nuclei)

    # stack the filtered masks
    cells_filtered_stack = np.stack((cells_filtered.copy(), intra_nuclei.copy()))
    stack_channels(name, masks_filtered, cells_filtered_stack)

# --------------- manually filter masks ---------------
# check output file to avoid unecessary re-validation
already_filtered_masks = [filename.replace('_mask.npy', '') for filename in os.listdir(
    f'{output_folder}') if '_mask.npy' in filename]

# move forward with new images
unval_images = dict([(key, val) for key, val in images.items()
                    if key not in already_filtered_masks])

# launch napari gui
filtered_masks = {}
for image_name, image_stack in unval_images.items():
    image_stack
    mask_stack = masks_filtered[image_name].copy()
    filtered_masks[image_name] = filter_masks(
        image_stack, image_name, mask_stack)

# --------------- process filtered masks ---------------
# TODO make below lines a new script
# reload previous masks for per-cell extraction
filtered_masks = {masks.replace('_mask.npy', ''): np.load(
    f'{output_folder}{masks}', allow_pickle=True) for masks in os.listdir(f'{output_folder}') if '_mask.npy' in masks}

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

# ---------------save arrays---------------
np.save(f'{output_folder}cytoplasm_masks.npy', cytoplasm_masks)
logger.info('mask arrays saved')
