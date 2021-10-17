from sklearn.model_selection import train_test_split
import numpy as np


def get_series_label(df, id_col, target_col):
    """
    Gets the label for a series
    Parameters
    ----------
    df: object
        dataframe from which to extract labels
    id_col: str
        column to group by
    target_col: str
        label column

    Returns
    -------
    For each group whether at least one label was positive or not
    """
    return df.groupby(id_col, sort=False).apply(lambda group: (group[target_col] == 1).any())


def split_data(data, id_col, target_col, train_size, random_state=0):
    """
    Get the ids for train and test set from a dataframe
    Parameters
    ----------
    data: object
        dataframe from which to extract the data sets
    id_col: str
        column to group by
    target_col: str
        label column
    train_size: float
        percentage of train set
    random_state: int
        random seed
    Returns
    -------
    Train and test set ids
    """
    print('Starting to prepare data')
    ids = data[id_col].unique()
    y = get_series_label(data, id_col, target_col)
    train_ids, test_ids = train_test_split(ids, train_size=train_size, random_state=random_state, stratify=y)

    return train_ids, test_ids


def normalize_data(train_data, test_data):
    """
    Standardizes train and test data
    Parameters
    ----------
    train_data:
        The training data
    test_data:
        The test data

    Returns
    -------
    Normalized data
    """
    means = train_data.mean(axis=0)
    stds = train_data.std(axis=0)
    stds[stds == 0] = 1

    train_data = (train_data - means) / stds
    test_data = (test_data - means) / stds

    return train_data, test_data


def balance_data_set(id_col, train_data, n_targets, undersample=True, imbalance=1.5):
    """
    Balances data set by under or oversampling
    Parameters
    ----------
    id_col: str
        column to group by
    train_data: object
        training data
    n_targets: int
        number of targets
    undersample: bool
        whether to under or oversample
    imbalance: float
        amount of imbalance to keep

    Returns
    -------
    Balanced data
    """
    # Get the number of patients with at least one positive day for each target then take the sum
    patients_grouped = train_data.groupby(id_col)
    n_pos_per_patient = patients_grouped.agg('sum').iloc[:, -n_targets:]
    n_pos_per_patient = n_pos_per_patient - patients_grouped.first().iloc[:, -n_targets:]

    if undersample:
        # Take the biggest value of all possible minority counts
        n_rows = (n_pos_per_patient != 0).sum(axis=0).argmax()
    else:
        # Take the smallest value of all possible minority counts
        n_rows = (n_pos_per_patient != 0).sum(axis=0).argmin()

    pos_ids = np.array(n_pos_per_patient[n_pos_per_patient.iloc[:, n_rows] != 0].index)
    neg_rows = n_pos_per_patient[n_pos_per_patient.iloc[:, n_rows] == 0].sum(axis=1)
    neg_pos_ids = np.array(neg_rows[neg_rows != 0].index)  # Neg in i_target but pos in other targets
    neg_ids = np.array(neg_rows[neg_rows == 0].index)

    np.random.shuffle(pos_ids)
    np.random.shuffle(neg_pos_ids)
    np.random.shuffle(neg_ids)

    print(len(neg_pos_ids), len(neg_ids))
    neg_ids = np.concatenate([neg_pos_ids, neg_ids])

    if pos_ids.shape[0] < neg_ids.shape[0]:
        minority_class = pos_ids
        majority_class = neg_ids
    else:
        minority_class = neg_ids
        majority_class = pos_ids
    minority_length = minority_class.shape[0]

    if undersample:
        total_ids = np.hstack([minority_class[:minority_length], majority_class[:int(minority_length * imbalance)]])
        np.random.shuffle(total_ids)
        train_data = train_data[train_data[id_col].isin(total_ids)]
        print(f'{n_rows=} {len(pos_ids)=} - {len(neg_ids)=} - {len(total_ids)=}')
        print('Balanced training data by undersampling')
    else:
        difference = majority_class.shape[0] // minority_length
        minority_data = train_data[train_data[id_col].isin(minority_class)]
        for i in range(difference - 1):
            train_data = train_data.append(minority_data)
        print(f'Added minority class {difference - 1} times')
        print('Balanced training data by oversampling')
    return train_data
