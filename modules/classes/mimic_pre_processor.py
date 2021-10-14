import os

import numpy as np
import pandas as pd

from modules.utils.handle_datasets import normalize_data, balance_data_set, split_data
from modules.utils.pad_sequences import pad_sequences, filter_sequences
from modules.utils.handle_directories import dump_pickle, get_pickle_file_path, load_pickle


def wbc_criterion(x):
    return (x > 12 or x < 4) and x != 0


def temp_criterion(x):
    return (x > 100.4 or x < 96.8) and x != 0


def save_data_to_disk(whole_data, mask, name, labels, output_folder, n_targets=1):
    """
    Persist data to disk with pickle
    Parameters
    ----------
    whole_data: object
        dataset to be persisted
    mask: object
        boolean mask of which entry is padded
    name: str
        filenames
    labels: list[str]
        target column(s)
    output_folder:
        target folder for saved files
    n_targets: int
        number of targets
    """
    # Because the targets are for the same day shift the targets by one and ignore the last day
    # because no targets exist
    input_data = whole_data[:, :-1, :-n_targets]
    targets = whole_data[:, 1:, -n_targets:]
    targets = targets.reshape(targets.shape[0], targets.shape[1], n_targets)
    input_data_mask = mask[:, :-1, :-n_targets]
    targets_mask = mask[:, 1:, -n_targets:]
    targets_mask = targets_mask.reshape(targets_mask.shape[0], targets_mask.shape[1], n_targets)

    assert input_data.shape == input_data_mask.shape
    assert targets.shape == targets_mask.shape

    n_pos = np.count_nonzero(targets.sum(axis=1), axis=0)
    print(name)
    print(f'Number of positive patients {n_pos}')
    print(f'Number of neg patients {whole_data.shape[0] - n_pos}')

    dump_pickle(input_data, get_pickle_file_path(f'{name}_data', labels, output_folder))
    dump_pickle(targets, get_pickle_file_path(f'{name}_targets', labels, output_folder))
    dump_pickle(input_data_mask, get_pickle_file_path(f'{name}_data_mask', labels, output_folder))
    dump_pickle(targets_mask, get_pickle_file_path(f'{name}_targets_mask', labels, output_folder))


class MimicPreProcessor(object):
    """
    Creates Data Sets for Machine learning from a parsed mimic file
    """

    def __init__(self, mimic_file_path, id_col='hadm_id', random_seed=0):
        self.parsed_mimic = pd.read_csv(mimic_file_path)
        self.id_col = id_col
        self.random_seed = random_seed

    def create_target(self, targets):
        """
        Given a dataframe creates a specified target for it as well as deleting columns that make the task trivial
        Parameters
        ----------
        targets: list
            An array of targets
        Returns
        -------
        dataframe with target column(s) as well as a list of feature names
        """
        df = self.parsed_mimic.copy()
        # Delete features that make the task trivial
        trivial_features = ['subject_id', 'yob', 'admityear', 'ct_angio', 'infection', 'ckd']
        if 'MI' in targets:
            df['MI'] = ((df['troponin'] > 0.4) & (df['ckd'] == 0)).apply(lambda x: int(x))
            trivial_features += ['troponin', 'troponin_std', 'troponin_min', 'troponin_max']
        if 'SEPSIS' in targets:
            hr_sepsis = df['heart rate'].apply(lambda x: 1 if x > 90 else 0)
            respiratory_rate_sepsis = df['respiratory rate'].apply(lambda x: 1 if x > 20 else 0)
            wbc_sepsis = df['wbcs'].apply(wbc_criterion)
            temperature_f_sepsis = df['temperature (f)'].apply(temp_criterion)
            sepsis_points = (hr_sepsis + respiratory_rate_sepsis + wbc_sepsis + temperature_f_sepsis)
            df['SEPSIS'] = ((sepsis_points >= 2) & (df['infection'] == 1)).apply(lambda x: int(x))
        if 'VANCOMYCIN' in targets:
            df['VANCOMYCIN'] = df['vancomycin'].apply(lambda x: 1 if x > 0 else 0)
            trivial_features += ['vancomycin']

        df = df.drop(trivial_features, axis=1, errors='ignore')
        df = df.select_dtypes(exclude=['object'])

        print(f'Created target {targets}')

        return df

    def create_time_col(self, df, n_time_steps):
        """
        Create a column that indicates the timestep each row is in
        Parameters
        ----------
        df: object
            dataframe to which to add the column
        n_time_steps: int
            number of timesteps
        """
        df.insert(0, 'group_index', df.groupby(self.id_col).cumcount() / (n_time_steps - 1))
        print(f'Created time column')
        return df

    def split_and_normalize_data(self, df, train_percentage, n_targets=1, key_path='./data/pickled_data_sets/'):
        """
        Splits data into train and test set. Then applies normalization
        Parameters
        ----------
        df: object
            data to be split into train and test set
        train_percentage: float
            percentage of training samples
        n_targets: int
            number of targets
        key_path: str
            path to key files

        Returns
        -------
        train and test set
        """
        print(f'{train_percentage=}')
        # Try loading train test split. If it does not exist make it yourself
        try:
            train_keys = load_pickle(f'{key_path}train_keys.pickle')
            test_keys = load_pickle(f'{key_path}test_keys.pickle')
        except IOError:
            train_keys, test_keys = split_data(df, self.id_col, df.columns[-1], train_percentage, random_state=0)
            dump_pickle(train_keys, f'{key_path}train_keys.pickle')
            dump_pickle(test_keys, f'{key_path}test_keys.pickle')
        train_data = df[df[self.id_col].isin(train_keys)]
        test_data = df[df[self.id_col].isin(test_keys)]

        # +1 to also exclude the id col
        train_subset = train_data.iloc[:, :-(n_targets + 1)]
        test_subset = test_data.iloc[:, :-(n_targets + 1)]
        train_subset, test_subset = normalize_data(train_subset, test_subset)
        train_data = pd.concat([train_subset, train_data.iloc[:, -(n_targets + 1):]], axis=1)
        test_data = pd.concat([test_subset, test_data.iloc[:, -(n_targets + 1):]], axis=1)

        print(f'{train_data.shape=} - {test_data.shape=}')

        if np.isnan(train_data).any().any():
            raise Exception('NaN Values remain in Train data')
        if np.isnan(test_data).any().any():
            raise Exception('NaN Values remain in Test data')

        return train_data, test_data

    def pad_data(self, df, time_steps, pad_value=0):
        """
        Pad dataframe and create boolean mask
        Parameters
        ----------
        df: object
            dataframe to be padded
        time_steps: int
            number of time steps to pad up to
        pad_value: float
            value with which the entry will get padded
        Returns
        -------
        padded data and boolean mask
        """
        df = pad_sequences(df, time_steps, pad_value=pad_value, grouping_col=self.id_col)
        df = df.drop(columns=[self.id_col])
        whole_data = df.values
        whole_data = whole_data.reshape(int(whole_data.shape[0] / time_steps), time_steps, whole_data.shape[1])

        # creating a second order bool matrix which keeps track of padded entries
        mask = (~whole_data.any(axis=2))
        whole_data[mask] = np.nan
        # restore 3D shape to boolmatrix for consistency
        mask = np.isnan(whole_data)
        whole_data[mask] = pad_value
        print("Padded data frame")
        return whole_data, mask

    def apply_pipeline(self, targets, n_time_steps, output_folder, balance_set=True):
        """
        A pipeline that:
            creates the target column target
            splits the data into train, validation and test set
            Padds the entry to n_time_steps
            Persists the data onto the disk to output_folder
        Parameters
        ----------
        targets: list
            target column(s)
        n_time_steps: int
            number of time steps
        output_folder: str
            target folder for saved files
        balance_set: bool
            whether to balance the data
        """
        n_targets = len(targets)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        df = self.create_target(targets)
        # df = self.create_time_col(df, n_time_steps)
        df = filter_sequences(df, 2, n_time_steps, grouping_col=self.id_col)

        train, test = self.split_and_normalize_data(df, train_percentage=0.8, n_targets=n_targets)
        if balance_set:
            train = balance_data_set(self.id_col, train, n_targets)
        feature_names = list(train.columns[:-n_targets])
        feature_names.remove(self.id_col)
        dump_pickle(feature_names, get_pickle_file_path('features', targets, output_folder))

        for dataset, name in [(train, 'train'), (test, 'test')]:
            whole_data, mask = self.pad_data(dataset, time_steps=n_time_steps)
            save_data_to_disk(whole_data, mask, name, targets, output_folder, n_targets=n_targets)

        print(f'Saved files to folder: {output_folder}')
