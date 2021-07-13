import pickle

import numpy as np
import torch
from torch.utils.data import TensorDataset


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


def get_pickle_folder(mimic_version, n_time_steps, seed=None):
    """
    @return: path to data for the experiments
    """
    path = f'./data/pickled_data_sets/mimic_{mimic_version}/{n_time_steps}_ts'
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


def return_loaded_model(model_name):
    """
    Loads a saved pytorch model from the disk
    @param model_name: name of the .h5 file
    @return: pytorch model
    """
    return torch.load("./output/models/best_models/{0}.h5".format(model_name))


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
