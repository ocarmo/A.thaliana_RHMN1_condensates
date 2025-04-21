import os
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from statannotations.Annotator import Annotator
from loguru import logger
plt.rcParams.update({'font.size': 14})
sns.set_palette('Paired')

logger.info('import ok')

input_folder = 'python_results/summary_calculations/'
output_folder = 'python_results/plotting/'

if not os.path.exists(output_folder):
    os.mkdir(output_folder)


# ---------------- initialise dataframes ----------------
# puncta features
puncta_features = pd.read_csv(f'{input_folder}puncta_features.csv')
puncta_features_reps = pd.read_csv(f'{input_folder}puncta_features_reps.csv')
puncta_features_normalized = pd.read_csv(f'{input_folder}puncta_features_normalized.csv')
puncta_features_normalized_reps = pd.read_csv(f'{input_folder}puncta_features_normalized_reps.csv')

# puncta features calculated per cell
puncta_features_percell = pd.read_csv(f'{input_folder}puncta_features_percell.csv')
puncta_features_percell_reps = pd.read_csv(f'{input_folder}puncta_features_percell_reps.csv')
puncta_features_percell_normalized = pd.read_csv(f'{input_folder}puncta_features_percell_normalized.csv')
puncta_features_percell_normalized_reps = pd.read_csv(f'{input_folder}puncta_features_percell_normalized_reps.csv')

# features of interest per dataframe
puncta_features_of_interest = ['puncta_area', 'puncta_eccentricity', 'puncta_aspect_ratio', 'puncta_circularity', 'puncta_cv', 'puncta_skew', 'g3bp_partition_coeff', 'rhm1_partition_coeff', 'cell_cv', 'cell_skew']
percell_features_of_interest = ['mean_puncta_area', 'puncta_area_proportion', 'puncta_count', 'puncta_mean_minor_axis', 'puncta_mean_major_axis', 'avg_eccentricity', 'puncta_cv_mean', 'puncta_skew_mean', 'g3bp_partition_coeff', 'rhm1_partition_coeff', 'cell_cv', 'cell_skew', 'cell_rhm1_intensity_mean']

# -------------- plotting data --------------
# suptitle, features_of_interest, raw_data, averaged_data, save_name
plotting_list = [
    ['per puncta, not normalized', puncta_features_of_interest, puncta_features, puncta_features_reps,'puncta-features_perpuncta_raw'],
    ['per puncta, normalized to cell intensity', puncta_features_of_interest, puncta_features_normalized, puncta_features_normalized_reps, 'puncta-features_perpuncta_normalized'],
    ['per cell, not normalized to cytoplasm intensity', percell_features_of_interest, puncta_features_percell, puncta_features_percell_reps, 'puncta-features_percell_raw'],
    ['per cell, normalized to cytoplasm intensity', percell_features_of_interest, puncta_features_percell_normalized, puncta_features_percell_normalized_reps, 'puncta-features_percell_normalized']
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
        ax = plt.subplot(5, 3, n + 1)

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


# plot by pairing conditions, no stats
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
        ax = plt.subplot(5, 3, n + 1)

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


# plot partitioning coefficients only
palette = ['#A6CEE3', '#1F78B4', '#F5CB5C']
order = ['PBS', 'NaAsO2', 'HS']
x = 'tag'
y = 'condition'
sublist = plotting_list[2]

raw_data = sublist[2]
raw_data = pd.melt(raw_data, id_vars=['image_name', 'tag', 'condition'], value_vars=['g3bp_partition_coeff', 'rhm1_partition_coeff'], var_name='channel', value_name='partition_coeff')

rep_data = sublist[3]
rep_data = pd.melt(rep_data, id_vars=['rep', 'tag', 'condition'], value_vars=['g3bp_partition_coeff', 'rhm1_partition_coeff'], var_name='channel', value_name='partition_coeff')

# create first FacetGrid with data from biological reps
g1 = sns.FacetGrid(rep_data, col='channel', height=4.5, aspect=0.8)
g1.map_dataframe(sns.boxplot, data=rep_data, x=x, y='partition_coeff',
                palette=['.9'], hue=y, hue_order=order, zorder=0)
g1.map_dataframe(sns.stripplot, data=rep_data, x=x, y='partition_coeff', dodge='True', 
                edgecolor='k', linewidth=1, hue=y, palette=palette, hue_order=order, zorder=2, size=8)
legend_labels = ['PBS', r'NaAsO$_{2}$', 'HS']
g1.add_legend(legend_data={key: value for key, value in zip(legend_labels, g1._legend_data.values())})

# create second FacetGrid with all data
g2 = sns.FacetGrid(raw_data, col='channel')

# iterate through the axes and plot the second dataset
for ax_i, category in enumerate(g2.col_names):
    ax = g1.axes.flat[ax_i]
    data_subset = raw_data[raw_data['channel'] == category]
    sns.stripplot(data=data_subset, x=x, y='partition_coeff', dodge='True',
                edgecolor='white', linewidth=1, alpha=0.4, hue=y, palette=palette, hue_order=order, zorder=1,size=8, ax=ax)
    ax.get_legend().remove()

# formatting
g1.set_xticklabels(['FLAG-RHM1', 'GFP-RHM1'])
g1.set_xlabels('')
g1.tight_layout()
g1.fig.savefig(f'{output_folder}condition-paired_{sublist[4]}_partition-only.png', bbox_inches='tight', pad_inches = 0.1, dpi = 300)