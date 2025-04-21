"""Collects images from the raw_data folder and organizes them into a [GBP1, RHM1, H342] np stack for later analyses
"""

from aicsimageio.writers import OmeTiffWriter
from aicsimageio import AICSImage
from loguru import logger
import numpy as np
import os

input_folder = 'raw_data'
output_folder = 'python_results/initial_cleanup/'

def czi_converter(image_name, input_folder, output_folder, tiff=False, array=True):
    """Stack images from nested .czi files and save for subsequent processing

    Args:
        image_name (str): image name (usually iterated from list)
        input_folder (str): filepath
        output_folder (str): filepath
        tiff (bool, optional): Save tiff. Defaults to False.
        array (bool, optional): Save np array. Defaults to True.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    # import image
    image = AICSImage(
            f'{input_folder}{image_name}.czi').get_image_data("CYX", B=0, Z=0, V=0, T=0)
    image_name = image_name.split('\\')[-1]

    if tiff == True:
        # save image to TIFF file with image number
        OmeTiffWriter.save(
            image, f'{output_folder}{image_name}.tif', dim_order='CYX')

    if array == True:
        np.save(f'{output_folder}{image_name}.npy', image)


# --------------- initalize file_list ---------------
source = 'raw_data'

if source == 'raw_data':
    flat_file_list = [filename for filename in os.listdir(input_folder) if '.czi' in filename]

else:
    # find directories of interest from shared drive 'M:/Olivia'
    experiments = ['2024-11-22', '2024-11-27', '2024-11-29']
    walk_list = [x[0] for x in os.walk(input_folder)]
    walk_list = [item for item in walk_list if any(x in item for x in experiments)]

    # read in all image files
    file_list = []
    for folder_path in walk_list:
        folder_path
        images = [[f'{root}\{filename}' for filename in files if '.czi' in filename]
            for root, dirs, files in os.walk(f'{folder_path}')]
        file_list.append(images[0])

    # flatten file_list
    flat_file_list = [item for sublist in file_list for item in sublist]

# remove images that do not require analysis (e.g., qualitative controls)
do_not_quantitate = ['_no-', 'UT'] 

# --------------- collect image names and convert ---------------
image_names = []
for filename in flat_file_list:
    if all(word not in filename for word in do_not_quantitate):
        filename
        filename = filename.split('.')[0]
        filename = filename.split(f'{input_folder}')[-1]
        image_names.append(filename)

# remove duplicates
image_names = list(dict.fromkeys(image_names))

# collect and convert images to np arrays
for name in image_names:
    czi_converter(name, input_folder=f'{input_folder}',
                  output_folder=f'{output_folder}')

logger.info('initial cleanup complete :)')
