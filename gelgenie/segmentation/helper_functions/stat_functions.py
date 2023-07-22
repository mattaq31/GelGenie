import os
import pandas as pd


def save_statistics(experiment_log_dir, filename, stats_dict, selected_data=None, append=True):

    true_filename = os.path.join(experiment_log_dir, filename)

    pd_data = pd.DataFrame.from_dict(stats_dict)

    if selected_data is not None and os.path.isfile(true_filename):
        if type(selected_data) == int:
            selected_data = [selected_data]
        pd_data = pd_data.loc[selected_data]

    if not os.path.isfile(true_filename):  # if there is no file in place, no point in appending
        append = False

    # TODO: the below can output numbers with too many DPs.  Need to either decide on good level of precision (e.g. 6 DP)
    # or ignore.  More details here: https://stackoverflow.com/questions/12877189/float64-with-pandas-to-csv
    pd_data.to_csv(true_filename, mode='a' if append else 'w', header=not append, index=False)


def load_statistics(experiment_log_dir, filename, config='dict'):
    stats = pd.read_csv(os.path.join(experiment_log_dir, filename))
    if config == 'dict':
        return stats.to_dict(orient='list')
    elif config == 'pd':
        return stats
