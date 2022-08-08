from pathlib import Path

import numpy as np
import pandas as pd


def excel_stats(train_loss_log, val_loss_log, base_dir):
    loss_array = np.array([train_loss_log, val_loss_log]).T
    loss_dataframe = pd.DataFrame(loss_array, columns=['Training Loss', 'Validation Dice Score'])
    loss_dataframe.index.names = ['Epoch']
    loss_dataframe.index += 1
    loss_dataframe.to_csv(Path(base_dir + '/loss.csv'))
