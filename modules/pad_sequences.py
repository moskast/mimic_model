''' Meant to deal with padding for an RNN when keras.preprocessing.pad_sequences fails '''

import numpy as np
import pandas as pd
import torch


def pad_sequences(df, lb, time_steps, pad_value, id_col='hadm_id'):
    ''' Takes a file path for the dataframe to operate on. lb is a lower bound to discard
        ub is an upper bound to truncate on. All entries are padded to their upper bound '''
    print('There are {0} rows in the df before padding'.format(len(df)))
    df = df.groupby(id_col).filter(lambda group: len(group) > lb).reset_index(drop=True)
    df = df.groupby(id_col).apply(lambda group: group[0:time_steps]).reset_index(drop=True)
    df = df.groupby(id_col).apply(lambda group: pd.concat(
        [group, pd.DataFrame(pad_value * np.ones((time_steps - len(group), len(df.columns))), columns=df.columns)],
        axis=0)).reset_index(drop=True)
    print('There are {0} rows in the df after padding'.format(len(df)))
    return df


def z_score_normalize(matrix):
    ''' Performs Z Score Normalization for 3rd order tensors
        matrix should be (batchsize, time_steps, features)
        Padded time steps should be masked with np.nan '''

    x_matrix = matrix[:, :, 0:-1]
    y_matrix = matrix[:, :, -1]
    y_matrix = y_matrix.reshape(y_matrix.shape[0], y_matrix.shape[1], 1)
    means = np.nanmean(x_matrix, axis=(0, 1))
    stds = np.nanstd(x_matrix, axis=(0, 1))
    x_matrix = x_matrix - means
    x_matrix = x_matrix / stds
    matrix = np.concatenate([x_matrix, y_matrix], axis=2)
    return matrix


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
