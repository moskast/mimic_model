import numpy as np
import pandas as pd
import torch


def filter_sequences(df, lower_bound, upper_bound, grouping_col='hadm_id'):
    """
    Filters sequences so that only those that are between lower_bound and upper_bound remain
    Parameters
    ----------
    df: object
        data on which to operate on
    lower_bound: int
        lower bound to discard
    upper_bound: int
        upper bound to truncate on
    grouping_col: str
        column to group by

    Returns
    -------
    sequences that are in between lower_bound and upper_bound
    """
    print("Start filtering procedure")
    df_grouped = df.groupby(grouping_col)
    if df_grouped.size().max() > upper_bound:
        df = df_grouped.apply(lambda group: group[:upper_bound]).reset_index(drop=True)
        print('Finished grouping')
    del df_grouped
    if lower_bound >= 1:
        df = df.groupby(grouping_col).filter(lambda group: len(group) > lower_bound).reset_index(drop=True)
        print('Finished filtering')
    return df


def pad_sequences(df, n_time_steps, pad_value, grouping_col='hadm_id'):
    """
    All entries are padded to their upper bound
    Parameters
    ----------
    df: object
        dataframe on which to operate on
    n_time_steps: int
        number of time_steps to pad up to
    pad_value: float
        value with which the entries get padded
    grouping_col: str
        column to group by

    Returns
    -------
    padded dataframe
    """
    print('There are {0} rows in the df before padding'.format(len(df)))
    df = df.groupby(grouping_col).apply(lambda group: pd.concat(
        [group, pd.DataFrame(pad_value * np.ones((n_time_steps - len(group), len(df.columns))), columns=df.columns)],
        axis=0)).reset_index(drop=True)
    print('There are {0} rows in the df after padding'.format(len(df)))
    return df


def get_seq_length_from_padded_seq(sequences):
    """
    Finds out pre padded length of a batch of sequences
    ----------
    sequences: object
        sequence from which original length should be inferred

    Returns
    -------
    original length list
    """
    max_len = sequences.shape[1]
    length_list = []
    for item in sequences:
        length = max_len
        indices = np.where(item.any(axis=1))[0]
        if len(indices) > 0:
            length = indices[-1] + 1
        length_list.append(length)
    return torch.tensor(length_list)
