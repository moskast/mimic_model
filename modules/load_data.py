import pickle

import numpy as np
import torch
from torch.utils.data import TensorDataset
import xgboost as xgb

from modules.config import AppConfig


def dump_pickle(variable, path):
    """
    Save a file via pickle
    @param variable: file to save
    @param path: path of the saved file
    """
    with open(path, 'wb') as file:
        pickle.dump(variable, file)


def load_pickle(path):
    """
    Load a saved pickle file
    @param path: path of the saved file
    @return: loaded pickle file
    """
    with open(path, 'rb') as file:
        variable = pickle.load(file)
    return variable


def get_output_directory(create_statistics=None, balance_data=None, oversample=None):
    if create_statistics is None:
        create_statistics = AppConfig.create_statistics
    if balance_data is None:
        balance_data = AppConfig.balance_data
    if oversample is None:
        oversample = AppConfig.oversample
    stat = 'st' if create_statistics else 'ns'
    bal = 'us' if balance_data else 'ub'
    bal = 'os' if oversample else bal
    return f'./output_{stat}_{bal}/'


def get_train_folders(create_statistics=None, balance_data=None, oversample=None):
    base_dir = get_output_directory(create_statistics, balance_data, oversample)
    model_dir = f'{base_dir}models/'
    checkpoint_dir = f'{model_dir}best_models/'
    final_model_dir = f'{model_dir}fully_trained_models/'
    logs_dir = f'{base_dir}logs'
    return checkpoint_dir, final_model_dir, logs_dir


def get_figure_dir(create_statistics=None, balance_data=None, oversample=None):
    base_dir = get_output_directory(create_statistics, balance_data, oversample)
    return base_dir + 'figures'


def get_pickle_folder(mimic_version, n_time_steps,
                      created_statistics=AppConfig.create_statistics, balanced=AppConfig.balance_data,
                      seed=AppConfig.random_seed):
    """
    @return: path to data for the experiments
    """

    stat = 'with_stat' if created_statistics else 'no_stat'
    balanced = 'b' if balanced else 'ub'
    path = f'./data/pickled_data_sets/mimic_{mimic_version}/{n_time_steps}_ts_{stat}_{balanced}'
    if seed is not None:
        path += f'_{seed}'
    return path


def get_pickle_file_path(file_name, target, folder='.output/pickled_data_sets'):
    """
    Returns a path to a saved file
    @param file_name: name of the saved file
    @param target: name of target column
    @param folder: name of the folder file is saved in
    @return: the path to the pickle folder with the filename structure {target}_{variable_name}.pickle in this case
    """
    return f'{folder}/{target}_{file_name}.pickle'


def load_model(model_name, path='./output/models/best_models/'):
    if 'xgb' in model_name.lower():
        model = load_xgb_model(path + model_name + '.model')
    else:
        model = load_pytorch_model(path + model_name + '.h5').cpu()
        model = model.eval()
    return model


def load_xgb_model(path):
    bst = xgb.Booster()
    bst.load_model(path)
    return bst


def load_pytorch_model(path):
    return torch.load(path)


def load_data_sets(data_path, target, n_percentage, reduce_dimensions=False):
    """
    Load data and convert it to Pytorch datasets optionally reduce the time dimension
    @param data_path: folder that contains the data files
    @param target: variable to predict
    @param n_percentage: percentage of full data to use
    @param reduce_dimensions: whether to reduce the time dimension
    @return: train data as Pytorch dataset as well as the number of features
    """
    training_data = load_pickle(get_pickle_file_path('train_data', target, data_path))
    training_targets = load_pickle(get_pickle_file_path('train_targets', target, data_path))

    # N_Samples x Seq_Length x N_Features
    training_data = training_data[0:int(n_percentage * training_data.shape[0])]  # Subsample if necessary
    training_targets = training_targets[0:int(n_percentage * training_targets.shape[0])]

    n_features = training_data.shape[-1]

    if reduce_dimensions:
        n_targets = training_targets.shape[-1]

        training_data = training_data.reshape(-1, n_features)  # Reshape to delete time dimension
        train_rows = ~np.all(training_data == 0, axis=1)
        training_data = training_data[train_rows]
        training_targets = training_targets.reshape(-1, n_targets)
        training_targets = training_targets[train_rows]

    train_dataset = TensorDataset(torch.tensor(training_data, dtype=torch.float),
                                  torch.tensor(training_targets, dtype=torch.float))

    return train_dataset, n_features
