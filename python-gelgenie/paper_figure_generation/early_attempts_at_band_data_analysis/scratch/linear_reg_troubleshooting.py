from pathlib import Path
import os
from os.path import join
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import re
from gelgenie.segmentation.helper_functions.general_functions import create_dir_if_empty, index_converter
from collections import defaultdict
import pickle
import math
from scipy.stats import linregress
import pingouin
plt.rcParams.update({'font.sans-serif':'Helvetica'})  # consistent figure formatting


def hidden_linreg(target, ref, num_hide=3, num_reps=10):
    # Main evaluation function for a gel lane -
    # hides a number of bands, attempts to linear fit the remaining bands and then predicts the value of the unseen bands

    # Initialize an empty list to store the selected sets
    selected_sets = set()

    # Loop to select unique sets
    for _ in range(num_reps):
        # Select the first num_hide numbers as a set
        selected_set = tuple(np.random.choice(range(len(target)), num_hide, replace=False))
        # Check if the set is already selected
        while selected_set in selected_sets:
            selected_set = tuple(np.random.choice(range(len(target)), num_hide, replace=False))
        selected_sets.add(selected_set)

    errors = []
    full_errors = []
    for combo in selected_sets:
        tfrac, rfrac = [t for ind, t in enumerate(target) if ind not in combo], [r for ind, r in enumerate(ref) if
                                                                                 ind not in combo]
        slope, intercept, r_value, p_value, std_err = linregress(tfrac, rfrac)
        pred_hidden = [slope * target[sel_ind] + intercept for sel_ind in combo]
        ref_hidden = [ref[sel_ind] for sel_ind in combo]
        percentage_errors = [(np.abs(t - p) / t) * 100 for t, p in zip(ref_hidden, pred_hidden)]
        errors.append(np.average(percentage_errors))
        full_errors.extend(percentage_errors)
    return errors, full_errors, (slope, intercept, combo)


def full_linreg(target, ref):
    # This function simply calculate linear regression on the full dataset, then computes the MAE and MAPE

    slope, intercept, r_value, p_value, std_err = linregress(target, ref)
    pred_values = [slope * t + intercept for t in target]
    errors = [np.abs(p - r) for p, r in zip(pred_values, ref)]
    percentage_errors = [(np.abs(r - p) / r) * 100 for p, r in zip(pred_values, ref)]
    return errors, percentage_errors, (slope, intercept)


# Full dataset loaded in here
data_pack = pickle.load(open('/Users/matt/Documents/PhD/research_output/Automatic_Gel_Analyzer/quantitative_evaluation/ladder_eval/combined_data_with_erosions_dilations_and_gelanalyzer.pkl','rb'))
for key,val in data_pack.items():
    data_pack[key]['Rectangularity'] = data_pack[key]['Pixel Count']/(data_pack[key]['Band Width']*data_pack[key]['Band Height'])
non_data_cols = ['Lane ID', 'Band ID', 'Rectangularity', 'Band Height', 'Band Width', 'Pixel Count', 'Ref.', 'Pixel STD']


# Additional GG-direct data loaded here (mainly to get global + local background correction)
def load_gg_csv_files_to_dict(folder_path, prefix="prefix_"):
    """
    Load CSV files from a folder and create a dictionary of DataFrames with prefixed keys.

    Parameters:
    - folder_path (str): The path to the folder containing CSV files.
    - prefix (str): The prefix to be added to the keys of the dictionary (default is "prefix_").

    Returns:
    - dataframes_dict (dict): A dictionary where keys are prefixed numbers, and values are corresponding DataFrames.
    """
    dataframes_dict = {}  # Dictionary to store DataFrames

    # Iterate through files in the folder
    for filename in os.listdir(folder_path):
        # Check if the file is a CSV file
        if filename.endswith(".csv"):
            # Extract the number from the filename
            file_number = filename.split("_")[0]

            # Read the CSV file into a DataFrame
            file_path = os.path.join(folder_path, filename)
            dataframe = pd.read_csv(file_path)

            # Add the prefix to the number and use it as the key in the dictionary
            key = f"{prefix}{file_number}"
            dataframes_dict[key] = dataframe
    return dataframes_dict

gg_path = Path("/Users/matt/Documents/PhD/research_output/Automatic_Gel_Analyzer/quantitative_evaluation/ladder_eval/qupath_data/james_data_v3_fixed_global/Data_with_norm_and_corrections")
gg_dfs = load_gg_csv_files_to_dict(gg_path, "gg_") # loads data and converts to dictionary
gg_dfs = {key: gg_dfs[key] for key in sorted(gg_dfs.keys(), key=lambda s: [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)])}  # sorts by gel ID


# combines GG-direct background corrected vals with the main data pack
for key, val in data_pack.items():
    gg_df = gg_dfs['gg_%s' % key.split('_')[0]][['Local Corrected Volume', 'Global Corrected Volume', 'Lane ID', 'Band ID']]
    data_pack[key] = pd.merge(val, gg_df, on=['Lane ID', 'Band ID'])

# linear regression for dataset carried out here
# currently, all data except lane 2 of 14,15,16 are considered (the lane was impacted by the high-temp gel running and bands are not accurate)
error_dict = defaultdict(list)
percentage_error_dict = defaultdict(list)
descriptor_dict = defaultdict(list)
high_intensity = ['3_Thermo', '4_Thermo', '6_Thermo', '7_Thermo', '8_Thermo', '10_Thermo', '11_Thermo', '12_Thermo',
                  '17_NEB', '19_NEB', '20_NEB', '21_NEB', '24_NEB', '29_NEB', '31_NEB', '32_Thermo', '33_NEB']
low_intensity = ['0_Thermo', '1_Thermo', '2_Thermo', '5_Thermo', '9_Thermo', '13_NEB', '14_NEB', '15_NEB', '16_NEB',
                 '18_NEB', '22_NEB', '23_NEB', '34_Thermo']

np.random.seed(12)

for big_key in data_pack.keys():
    sel_df = data_pack[big_key]
    ladder_type = 'NEB'
    if 'Thermo' in big_key:
        ladder_type = 'ThermoFisher'
    for col_index, column in enumerate(sel_df.columns):
        if column in non_data_cols:
            continue
        if ' BC' in column and 'Raw' not in column:
            continue
        for lane in np.unique(sel_df['Lane ID']):

            if lane == 2 and big_key in ['14_NEB', '16_NEB', '17_NEB']:
                continue

            ref = sel_df[sel_df['Lane ID'] == lane]['Ref.'].to_numpy()
            target = sel_df[sel_df['Lane ID'] == lane][column].to_numpy()
            errors, percentage_errors, (slope, intercept) = full_linreg(target, ref)

            error_dict[column].extend(errors)
            percentage_error_dict[column].extend(percentage_errors)
            if column == 'Raw':  # adds one time data for each lane e.g. rectangularity level, etc.
                rectangularity_sum = np.sum(sel_df[sel_df['Lane ID'] == lane]['Rectangularity'].to_numpy()) / len(ref)
                descriptor_dict['Rectangularity'].append(rectangularity_sum)

                pixel_av = np.sum(sel_df[sel_df['Lane ID'] == lane]['Pixel Average'].to_numpy()) / len(ref)
                if pixel_av > 255:  # 16-bit norm
                    pixel_av = pixel_av / 65535
                else:  # 8-bit norm
                    pixel_av = pixel_av / 255

                descriptor_dict['Pixel Average'].append(pixel_av)
                descriptor_dict['Gel Name'].append(big_key)
                error_dict['Ladder'].extend([ladder_type] * len(errors))

                if big_key in high_intensity:
                    descriptor_dict['Gel Intensity'].append('High')
                else:
                    descriptor_dict['Gel Intensity'].append('Low')

error_df = pd.DataFrame.from_dict(error_dict)
perc_error_df = pd.DataFrame.from_dict(percentage_error_dict)  # this isn't necessary, but nice to have
descriptor_df = pd.DataFrame.from_dict(descriptor_dict)
