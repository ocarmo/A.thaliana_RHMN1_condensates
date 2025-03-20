"""Applies cellpose algorithms to determine cellular and nuclear masks
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from cellpose import models
from cellpose import plot
from loguru import logger
from scipy import ndimage as ndi
from skimage import (
    filters, measure, morphology, segmentation
)
from cellpose.io import logger_setup
logger_setup();

input_folder = 'python_results/initial_cleanup/'
output_folder = 'python_results/cellpose_masking/'

if not os.path.exists(output_folder):
    os.mkdir(output_folder)


def apply_cellpose(images, image_type='cyto', channels = None, diameter=None, flow_threshold=0.4, cellprob_threshold=0.0, resample=False):
    """Apply standard cellpose model to list of images.

    Args:
        images (ndarray): numpy array of 16 bit images
        image_type (str, optional): Cellpose model. Defaults to 'cyto'.
        channels (int, optional): define CHANNELS to run segementation on (grayscale=0, R=1, G=2, B=3) where channels = [cytoplasm, nucleus]. Defaults to None.
        diameter (int, optional): Expected diameter of cell or nucleus. Defaults to None.
        flow_threshold (float, optional): maximum allowed error of the flows for each mask. Defaults to 0.4.
        cellprob_threshold (float, optional): The network predicts 3 outputs: flows in X, flows in Y, and cell “probability”. The predictions the network makes of the probability are the inputs to a sigmoid centered at zero (1 / (1 + e^-x)), so they vary from around -6 to +6. Decrease this threshold if cellpose is not returning as many ROIs as you expect. Defaults to 0.0.
        resample (bool, optional): Resampling can create smoother ROIs but take more time. Defaults to False.

    Returns:
        ndarray: array of masks, flows, styles, and diameters
    """
    if channels is None:
        channels = [0, 0]
    model = models.Cellpose(model_type=image_type)
    masks, flows, styles, diams = model.eval(
        images, diameter=diameter, channels=channels, flow_threshold=flow_threshold, cellprob_threshold=cellprob_threshold, resample=resample)
    return masks, flows, styles, diams


def visualise_cell_pose(images, masks, flows, channels = None):
    """Display cellpose results for each image

    Args:
        images (ndarray): single channel (one array)
        masks (ndarray): one array
        flows (_type_): _description_
        channels (_type_, optional): _description_. Defaults to None.
    """
    if channels is None:
        channels = [0, 0]
    for image_number, image in enumerate(images):
        maski = masks[image_number]
        flowi = flows[image_number][0]
        
        fig = plt.figure(figsize=(12, 5))
        plot.show_segmentation(fig, image, maski, flowi, channels=channels)
        plt.tight_layout()
        plt.show()


# ---------------- initialise file list ----------------
file_list = [filename for filename in os.listdir(
    input_folder) if 'npy' in filename]

imgs = [np.load(f'{input_folder}{filename}') for filename in file_list]

# ---------------- outline cells ----------------
# collecting only channel 0's for before
merged_chs = [(image[0] + image[2]) for image in imgs]
merged_chs_smooth = [filters.gaussian(image, sigma=5) for image in merged_chs]

# apply cellpose in chunks to save RAM
chunks_of_images = [merged_chs_smooth[i:i + 47] for i in range(0, len(merged_chs_smooth), 47)]
masks = []
for chunk in chunks_of_images:
    # apply cellpose
    chunk_masks, flows, styles, diams = apply_cellpose(chunk, image_type='cyto', diameter=600, flow_threshold=0.4, resample=True, cellprob_threshold=-1)
    masks.append(chunk_masks)
    # # visualize masking if needed
    # visualise_cell_pose(gfp_channel_bright_smooth[:8], flatten_masks, flows, channels=[0, 0])
flatten_masks = [item for sublist in masks for item in sublist]

# save cell masks before moving on to nuclei
np.save(f'{output_folder}cellpose_cellmasks.npy', flatten_masks)
logger.info('cell masks saved')

# ---------------- outline nuclei ----------------
# use hoechst to outline nucleus
nuc_images = [image[4] for image in imgs]

# apply cellpose in chunks to save RAM
nuc_chunks_of_images = [nuc_images[i:i + 47] for i in range(0, len(nuc_images), 47)]
nuc_masks = []
for chunk in nuc_chunks_of_images:
    # apply cellpose
    chunk_nuc_masks, nuc_flows, nuc_styles, nuc_diams = apply_cellpose(
        chunk, image_type='nuclei', diameter=400, resample=True, flow_threshold=0.2, cellprob_threshold=0)
    nuc_masks.append(chunk_nuc_masks)
flatten_nuc_masks = [item for sublist in nuc_masks for item in sublist]

# save nuclei masks
np.save(f'{output_folder}cellpose_nucmasks.npy', flatten_nuc_masks)
