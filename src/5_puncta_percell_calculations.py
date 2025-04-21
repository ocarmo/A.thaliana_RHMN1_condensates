import os
import numpy as np
import pandas as pd
import functools
from scipy import ndimage
from scipy import stats
from loguru import logger

logger.info('import ok')

input_folder = 'python_results/summary_calculations/'
output_folder = 'python_results/summary_calculations/'

# ---------------- initialise dataframe ----------------
feature_information = pd.read_csv(f'{input_folder}puncta_features.csv')

# -------------- calculate feature information per cell --------------
# grab average major and minor puncta axis length per cell 
minor_axis = feature_information.groupby(
    ['image_name', 'cell_number'])['puncta_minor_axis_length'].mean()
major_axis = feature_information.groupby(
    ['image_name', 'cell_number'])['puncta_major_axis_length'].mean()

# calculate average size of puncta per cell
puncta_avg_area = feature_information.groupby(
    ['image_name', 'cell_number'])['puncta_area'].mean().reset_index()

# calculate percent of cytoplasm area taken by puncta
cell_size = feature_information.groupby(
    ['image_name', 'cell_number'])['cell_size'].mean()
puncta_area = feature_information.groupby(
    ['image_name', 'cell_number'])['puncta_area'].sum()
puncta_proportion = ((puncta_area / cell_size) *
                   100).reset_index().rename(columns={0: 'proportion_puncta_area'})

# calculate number of puncta per cell
puncta_count = feature_information.groupby(
    ['image_name', 'cell_number'])['puncta_area'].count()

# grab average size of puncta per cell
avg_eccentricity = feature_information.groupby(
    ['image_name', 'cell_number'])['puncta_eccentricity'].mean().reset_index()

# grab average puncta coefficient of variance per cell
puncta_cv_mean = feature_information.groupby(
    ['image_name', 'cell_number'])['puncta_cv'].mean()

# grab average puncta intensity skew per cell 
puncta_skew_mean = feature_information.groupby(
    ['image_name', 'cell_number'])['puncta_skew'].mean()

# grab cell intensity mean 
cell_rhm1_intensity_mean = feature_information.groupby(
    ['image_name', 'cell_number'])['cell_rhm1_intensity_mean'].mean()

# grab cell g3bp1 partitioning coefficient (partitioning into rhm1 granules)
g3bp_partition_coeff = feature_information.groupby(
    ['image_name', 'cell_number'])['g3bp_partition_coeff'].mean()

# grab cell rhm1 partitioning coefficient (partitioning into rhm1 granules)
rhm1_partition_coeff = feature_information.groupby(
    ['image_name', 'cell_number'])['rhm1_partition_coeff'].mean()

# grab cell cell rhm1 coefficient of variance 
cell_cv = feature_information.groupby(
    ['image_name', 'cell_number'])['cell_cv'].mean()

# grab cell cell rhm1 skew 
cell_skew = feature_information.groupby(
    ['image_name', 'cell_number'])['cell_skew'].mean()

# summarise
summary = functools.reduce(lambda left, right: pd.merge(left, right, on=['image_name', 'cell_number'], how='outer'), [cell_size.reset_index(), puncta_avg_area, puncta_proportion, puncta_count.reset_index(), minor_axis, major_axis, avg_eccentricity, puncta_cv_mean, puncta_skew_mean, g3bp_partition_coeff, rhm1_partition_coeff, cell_cv, cell_skew, cell_rhm1_intensity_mean])
summary.columns = ['image_name', 'cell_number',  'cell_size', 'mean_puncta_area', 'puncta_area_proportion', 'puncta_count', 'puncta_mean_minor_axis', 'puncta_mean_major_axis', 'avg_eccentricity', 'puncta_cv_mean', 'puncta_skew_mean', 'g3bp_partition_coeff', 'rhm1_partition_coeff', 'cell_cv', 'cell_skew', 'cell_rhm1_intensity_mean']

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
summary.to_csv(f'{output_folder}puncta_features_percell.csv')

# average data by biological replicate
percell_reps = []
for col in cell_features_of_interest:
    reps_table = summary.groupby(['condition', 'tag', 'rep']).mean(numeric_only=True)[f'{col}']
    percell_reps.append(reps_table)
percell_reps_df = functools.reduce(lambda left, right: pd.merge(left, right, on=['condition', 'tag', 'rep'], how='outer'), percell_reps).reset_index()
percell_reps_df.to_csv(f'{output_folder}puncta_features_percell_reps.csv')

# normalize to mean cell intensity
normalized_summary = summary.copy()
for col in cell_features_of_interest:
    normalized_summary[col] = normalized_summary[col] / normalized_summary['cell_rhm1_intensity_mean']
normalized_summary.to_csv(f'{output_folder}puncta_features_percell_normalized.csv')

# average normalized data by biological replicate
normalized_summary_reps = []
for col in cell_features_of_interest:
    reps_table = normalized_summary.groupby(['condition', 'tag', 'rep']).mean(numeric_only=True)[f'{col}']
    normalized_summary_reps.append(reps_table)
normalized_summary_reps_df = functools.reduce(lambda left, right: pd.merge(left, right, on=['condition', 'tag', 'rep'], how='outer'), normalized_summary_reps).reset_index()
normalized_summary_reps_df.to_csv(f'{output_folder}puncta_features_percell_normalized_reps.csv')

logger.info('saved puncta feature averaged-per-cell dataframes')