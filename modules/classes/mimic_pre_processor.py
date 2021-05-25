import os

import numpy as np
import pandas as pd

from modules.pad_sequences import pad_sequences, filter_sequences
from modules.pickle_utils import dump_pickle, get_pickle_path


def wbc_criterion(x):
    return (x > 12 or x < 4) and x != 0


def temp_criterion(x):
    return (x > 100.4 or x < 96.8) and x != 0


class MimicPreProcessor(object):
    """
    Creates Data Sets for Machine learning from a parsed mimic file
    """

    def __init__(self, mimic_file_path, id_col='hadm_id', random_seed=42):
        self.parsed_mimic = pd.read_csv(mimic_file_path)
        self.id_col = id_col
        self.random_seed = random_seed

    def create_target(self, target):
        """
        Given a dataframe creates a specified target for it as well as deleting columns that make the task trivial
        @param target: One out of three targets to create [MI, SEPSIS, VANCOMYCIN]
        @return: dataframe with target column as well as a list of feature names
        """
        df = self.parsed_mimic.copy()
        # Delete features that make the task trivial
        toss = ['subject_id', 'yob', 'admityear']
        if target == 'MI':
            df[target] = ((df['troponin'] > 0.4) & (df['ckd'] == 0)).apply(lambda x: int(x))
            toss += ['ct_angio', 'troponin', 'troponin_std', 'troponin_min', 'troponin_max', 'infection', 'ckd']
        elif target == 'SEPSIS':
            df['hr_sepsis'] = df['heart rate'].apply(lambda x: 1 if x > 90 else 0)
            df['respiratory rate_sepsis'] = df['respiratory rate'].apply(lambda x: 1 if x > 20 else 0)
            df['wbc_sepsis'] = df['wbcs'].apply(wbc_criterion)
            df['temperature f_sepsis'] = df['temperature (f)'].apply(temp_criterion)
            df['sepsis_points'] = (df['hr_sepsis'] + df['respiratory rate_sepsis']
                                   + df['wbc_sepsis'] + df['temperature f_sepsis'])
            df[target] = ((df['sepsis_points'] >= 2) & (df['infection'] == 1)).apply(lambda x: int(x))
            del df['hr_sepsis']
            del df['respiratory rate_sepsis']
            del df['wbc_sepsis']
            del df['temperature f_sepsis']
            del df['sepsis_points']
            del df['infection']
            toss += ['ct_angio', 'infection', 'ckd']
        elif target == 'VANCOMYCIN':
            df[target] = df['vancomycin'].apply(lambda x: 1 if x > 0 else 0)
            del df['vancomycin']
            toss += ['ct_angio', 'infection', 'ckd']

        df = df.select_dtypes(exclude=['object'])
        feature_names = [i for i in list(df.columns) if i not in toss]
        df = df[feature_names]
        feature_names.remove(target)

        # Only remove id_col from features because it is needed later
        feature_names.remove(self.id_col)
        print(f'Created target {target}')

        return df, feature_names

    def split_and_normalize_data(self, df, train_percentage=0.7, val_percentage=0.1, undersample=True):
        """
        Splits data into train, validation and test set. Then applies normalization as well as undersampling (if specified)
        @param df: data to be split
        @param train_percentage: percentage of training samples
        @param val_percentage: percentage of validation samples
        @param undersample: whether to apply undersampling
        @return: train, validation and test set
        """
        keys = df[self.id_col].sample(frac=1).unique()
        train_bound = int(train_percentage * len(keys))
        val_bound = int((train_percentage + val_percentage) * len(keys))
        train_keys = keys[:train_bound]
        val_keys = keys[train_bound:val_bound]
        test_keys = keys[val_bound:]
        train_data = df[df[self.id_col].isin(train_keys)]
        val_data = df[df[self.id_col].isin(val_keys)]
        test_data = df[df[self.id_col].isin(test_keys)]

        if undersample:
            positive_rows = train_data.iloc[:, -1] == 1
            pos_ids = np.unique(train_data[positive_rows][self.id_col])
            np.random.shuffle(pos_ids)
            # Because the first check checks for positive labels per day most ids will also show up here
            # That is why the second term is needed
            neg_ids = np.unique(train_data[~positive_rows & ~(train_data[self.id_col].isin(pos_ids))][self.id_col])
            np.random.shuffle(neg_ids)
            length = min(pos_ids.shape[0], neg_ids.shape[0])
            total_ids = np.hstack([pos_ids[0:length], neg_ids[0:length]])
            np.random.shuffle(total_ids)
            train_data = df[df[self.id_col].isin(total_ids)]
            print('Balanced training data by undersampling')

        means = train_data.iloc[:, :-1].mean(axis=0)
        stds = train_data.iloc[:, :-1].std(axis=0)
        train_data.iloc[:, :-1] = (train_data.iloc[:, :-1] - means) / stds
        val_data.iloc[:, :-1] = (val_data.iloc[:, :-1] - means) / stds
        test_data.iloc[:, :-1] = (test_data.iloc[:, :-1] - means) / stds
        return train_data, val_data, test_data

    def pad_data(self, df, time_steps, pad_value=0):
        """
        Pad dataframe and create boolean mask
        @param df: dataframe to be padded
        @param time_steps: number of time steps to pad up to
        @param pad_value: value with which the entry will get padded
        @return: padded data and boolean mask
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

    def save_data_to_disk(self, whole_data, mask, name, target, output_folder):
        """
        Persist data to disk with pickle
        @param whole_data: dataset to be persisted
        @param mask: boolean mask of which entry is padded
        @param name: name of the files
        @param target: target column
        @param output_folder: target folder for saved files
        """
        # Because the targets are for the same day shift the targets by one and ignore the last day
        # because no targets exist
        input_data = whole_data[:, :-1, :-1]
        targets = whole_data[:, 1:, -1]
        targets = targets.reshape(targets.shape[0], targets.shape[1], 1)
        input_data_mask = mask[:, :-1, :-1]
        targets_mask = mask[:, 1:, -1]
        targets_mask = targets_mask.reshape(targets_mask.shape[0], targets_mask.shape[1], 1)

        assert input_data.shape == input_data_mask.shape
        assert targets.shape == targets_mask.shape

        dump_pickle(input_data, get_pickle_path(f'{name}_data', target, output_folder))
        dump_pickle(targets, get_pickle_path(f'{name}_targets', target, output_folder))
        dump_pickle(input_data_mask, get_pickle_path(f'{name}_data_mask', target, output_folder))
        dump_pickle(targets_mask, get_pickle_path(f'{name}_targets_mask', target, output_folder))

    def pre_process_and_save_files(self, target, n_time_steps, output_folder):
        """
        Run a pipeline that:
            creates the target column target
            splits the data into train, validation and test set
            Padds the entry to n_time_steps
            Persists the data onto the disk to output_folder
        @param target: target column
        @param n_time_steps: number of time steps
        @param output_folder: target folder for saved files
        """
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        df, feature_names = self.create_target(target)
        dump_pickle(feature_names, get_pickle_path('features', target, output_folder))
        df = filter_sequences(df, 2, n_time_steps, grouping_col=self.id_col)
        train, val, test = self.split_and_normalize_data(df, train_percentage=0.7, val_percentage=0.1, undersample=True)

        for dataset, name in [(train, 'train'), (val, 'validation'), (test, 'test')]:
            whole_data, mask = self.pad_data(dataset, time_steps=n_time_steps)
            self.save_data_to_disk(whole_data, mask, name, target, output_folder)

        print(f'Saved files to folder: {output_folder}')
