import numpy as np
import pandas as pd
import torch


def filter_sequences(df, lb, time_steps, grouping_col='hadm_id'):
    """
    Filters sequences so that only those that are between lb and time_steps remain
    @param df: dataframe on which to operate on
    @param lb: is a lower bound to discard
    @param time_steps: is an upper bound to truncate on
    @param grouping_col: grouping column
    @return: sequences that are in between lb and time_steps
    """
    print("Start filtering procedure")
    df_grouped = df.groupby(grouping_col)
    if df_grouped.size().max() > time_steps:
        df = df_grouped.apply(lambda group: group[:time_steps]).reset_index(drop=True)
        print('Finished grouping')
    del df_grouped
    if lb >= 1:
        df = df.groupby(grouping_col).filter(lambda group: len(group) > lb).reset_index(drop=True)
        print('Finished filtering')
    return df


def pad_sequences(df, time_steps, pad_value, grouping_col='hadm_id'):
    """
    All entries are padded to their upper bound
    @param df: dataframe on which to operate on
    @param time_steps: number of time_steps to pad up to
    @param pad_value: value with which the entries get padded
    @param grouping_col: grouping column
    @return: padded dataframe
    """
    print('There are {0} rows in the df before padding'.format(len(df)))
    df = df.groupby(grouping_col).apply(lambda group: pd.concat(
        [group, pd.DataFrame(pad_value * np.ones((time_steps - len(group), len(df.columns))), columns=df.columns)],
        axis=0)).reset_index(drop=True)
    print('There are {0} rows in the df after padding'.format(len(df)))
    return df


def get_seq_length_from_padded_seq(sequences):
    """
    Finds out pre padded length of a batch of sequences
    @param sequences: sequence from which original length should be inferred
    @return: original length list
    """
    max_len = sequences.shape[1]
    length_list = []
    for item in sequences:
        length = max_len
        indices = np.where(~item.any(axis=1))[0]
        if len(indices) > 0:
            length = indices[0]
        length_list.append(length)
    return torch.tensor(length_list)
