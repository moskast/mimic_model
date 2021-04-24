''' Meant to deal with padding for an RNN when keras.preprocessing.pad_sequences fails '''

import numpy as np
import pandas as pd
import torch


def filter_sequences(df, lb, time_steps, id_col='hadm_id'):
    print("Start filtering procedure")
    df_grouped = df.groupby(id_col)
    if df_grouped.size().max() > time_steps:
        df = df_grouped.apply(lambda group: group[:time_steps]).reset_index(drop=True)
        print('Finished grouping')
    del df_grouped
    if lb >= 1:
        df = df.groupby(id_col).filter(lambda group: len(group) > lb).reset_index(drop=True)
        print('Finished filtering')
    return df


def pad_sequences(df, time_steps, pad_value, id_col='hadm_id'):
    ''' Takes a file path for the dataframe to operate on. lb is a lower bound to discard
        ub is an upper bound to truncate on. All entries are padded to their upper bound '''
    print('There are {0} rows in the df before padding'.format(len(df)))
    df = df.groupby(id_col).apply(lambda group: pd.concat(
        [group, pd.DataFrame(pad_value * np.ones((time_steps - len(group), len(df.columns))), columns=df.columns)],
        axis=0)).reset_index(drop=True)
    print('There are {0} rows in the df after padding'.format(len(df)))
    return df


def get_seq_length_from_padded_seq(sequence):
    max_len = sequence.shape[1]
    length_list = []
    for item in sequence:
        length = max_len
        indices = np.where(~item.any(axis=1))[0]
        if len(indices) > 0:
            length = indices[0]
        length_list.append(length)
    return torch.tensor(length_list)
