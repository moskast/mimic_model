import os

import numpy as np
import pandas as pd

from modules.pad_sequences import pad_sequences, z_score_normalize, filter_sequences
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

    def pad_and_format_data(self, df, time_steps=14, pad_value=0, lower_bound=1):
        df = filter_sequences(df, 1, time_steps, id_col=self.id_col)
        #df = pad_sequences(df, time_steps, pad_value=pad_value, id_col=self.id_col)
        df = df.drop(columns=[self.id_col])
        whole_data = df.values
        whole_data = whole_data.reshape(int(whole_data.shape[0] / time_steps), time_steps, whole_data.shape[1])

        # creating a second order bool matrix which keeps track of padded entries
        mask = (~whole_data.any(axis=2))
        whole_data[mask] = np.nan
        whole_data = z_score_normalize(whole_data)
        # restore 3D shape to boolmatrix for consistency
        mask = np.isnan(whole_data)
        whole_data[mask] = pad_value
        print("Padded and formatted data frame")
        return whole_data, mask

    def create_learning_sets(self, whole_data, mask, train_percentage=0.7, val_percentage=0.1, undersample=True):
        np.random.seed(self.random_seed)
        val_percentage += train_percentage

        n_datapoints = whole_data.shape[0]
        permutation = np.random.permutation(n_datapoints)
        whole_data = whole_data[permutation]
        mask = mask[permutation]

        print(whole_data.shape)
        input_data = whole_data[:, :-1, 0:-1]
        targets = whole_data[:, 1:, -1]
        input_data_mask = mask[:, :-1, 0:-1]
        targets_mask = mask[:, 1:, -1]

        assert input_data.shape == input_data_mask.shape
        assert targets.shape == targets_mask.shape
        print(input_data.shape)
        print(targets.shape)

        data_train = input_data[0:int(train_percentage * n_datapoints)]
        targets_train = targets[0:int(train_percentage * n_datapoints)]
        targets_train = targets_train.reshape(targets_train.shape[0], targets_train.shape[1], 1)
        data_train_mask = input_data_mask[0:int(train_percentage * n_datapoints)]
        targets_train_mask = targets_mask[0:int(train_percentage * n_datapoints)]
        targets_train_mask = targets_train_mask.reshape(targets_train_mask.shape[0], targets_train_mask.shape[1], 1)

        data_validation = input_data[int(train_percentage * n_datapoints):int(val_percentage * n_datapoints)]
        target_validation = targets[int(train_percentage * n_datapoints):int(val_percentage * n_datapoints)]
        target_validation = target_validation.reshape(target_validation.shape[0], target_validation.shape[1], 1)
        data_validation_mask = input_data_mask[int(train_percentage * n_datapoints):int(val_percentage * n_datapoints)]
        targets_validation_mask = targets_mask[int(train_percentage * n_datapoints):int(val_percentage * n_datapoints)]
        targets_validation_mask = targets_validation_mask.reshape(targets_validation_mask.shape[0],
                                                                  targets_validation_mask.shape[1], 1)

        data_test = input_data[int(val_percentage * n_datapoints)::]
        targets_test = targets[int(val_percentage * n_datapoints)::]
        targets_test = targets_test.reshape(targets_test.shape[0], targets_test.shape[1], 1)
        data_test_mask = input_data_mask[int(val_percentage * n_datapoints)::]
        targets_test_mask = targets_mask[int(val_percentage * n_datapoints)::]
        targets_test_mask = targets_test_mask.reshape(targets_test_mask.shape[0], targets_test_mask.shape[1], 1)

        print("Created Matrices")

        if undersample:
            whole_data_train = np.concatenate([data_train, targets_train], axis=2)
            pos_ind = np.unique(np.where((whole_data_train[:, :, -1] == 1).any(axis=1))[0])
            np.random.shuffle(pos_ind)
            neg_ind = np.unique(np.where(~(whole_data_train[:, :, -1] == 1).any(axis=1))[0])
            np.random.shuffle(neg_ind)
            length = min(pos_ind.shape[0], neg_ind.shape[0])
            total_ind = np.hstack([pos_ind[0:length], neg_ind[0:length]])
            np.random.shuffle(total_ind)
            data_train = whole_data_train[total_ind, :, 0:-1]
            targets_train = whole_data_train[total_ind, :, -1]
            targets_train = targets_train.reshape(targets_train.shape[0], targets_train.shape[1], 1)

            data_train_mask = data_train_mask[total_ind]
            targets_train_mask = targets_train_mask[total_ind]
            print('Balanced training data')

        return (data_train, targets_train, data_train_mask, targets_train_mask,
                data_validation, target_validation, data_validation_mask, targets_validation_mask,
                data_test, targets_test, data_test_mask, targets_test_mask)

    def pre_process_and_save_files(self, target, time_steps, output_folder):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        df, feature_names = self.create_target(target)
        whole_data, mask = self.pad_and_format_data(df, time_steps=time_steps, lower_bound=5)
        (data_train, targets_train, data_train_mask, targets_train_mask,
         data_validation, target_validation, data_validation_mask, targets_validation_mask,
         data_test, targets_test, data_test_mask, targets_test_mask) = self.create_learning_sets(whole_data, mask)

        dump_pickle(feature_names, get_pickle_path('features', target, output_folder))

        dump_pickle(data_train, get_pickle_path('train_data', target, output_folder))
        dump_pickle(targets_train, get_pickle_path('train_targets', target, output_folder))
        dump_pickle(data_train_mask, get_pickle_path('train_data_mask', target, output_folder))
        dump_pickle(targets_train_mask, get_pickle_path('train_targets_mask', target, output_folder))

        dump_pickle(data_validation, get_pickle_path('validation_data', target, output_folder))
        dump_pickle(target_validation, get_pickle_path('validation_target', target, output_folder))
        dump_pickle(data_validation_mask, get_pickle_path('validation_data_mask', target, output_folder))
        dump_pickle(targets_validation_mask, get_pickle_path('validation_targets_mask', target, output_folder))

        dump_pickle(data_test, get_pickle_path('test_data', target, output_folder))
        dump_pickle(targets_test, get_pickle_path('test_targets', target, output_folder))
        dump_pickle(data_test_mask, get_pickle_path('test_data_mask', target, output_folder))
        dump_pickle(targets_test_mask, get_pickle_path('test_targets_mask', target, output_folder))
        print(f'Saved files to folder: {output_folder}')
