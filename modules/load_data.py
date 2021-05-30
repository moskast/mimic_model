import numpy as np
import torch
from torch.utils.data import TensorDataset

from modules.pickle_utils import load_pickle, get_pickle_path


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
    @return: train and validation data as Pytorch datasets as well as the number of features
    """
    training_data = load_pickle(get_pickle_path('train_data', target, data_path))
    training_targets = load_pickle(get_pickle_path('train_targets', target, data_path))
    validation_data = load_pickle(get_pickle_path('validation_data', target, data_path))
    validation_targets = load_pickle(get_pickle_path('validation_targets', target, data_path))

    # N_Samples x Seq_Length x N_Features
    training_data = training_data[0:int(n_percentage * training_data.shape[0])]  # Subsample if necessary
    training_targets = training_targets[0:int(n_percentage * training_targets.shape[0])]

    n_features = training_data.shape[2]

    if reduce_dimensions:
        training_data = training_data.reshape(-1, n_features)  # Reshape to delete time dimension
        train_rows = ~np.all(training_data == 0, axis=1)
        training_data = training_data[train_rows]
        training_targets = training_targets.reshape(-1, 1)
        training_targets = training_targets[train_rows]

        validation_data = validation_data.reshape(-1, n_features)
        validation_rows = ~np.all(validation_data == 0, axis=1)
        validation_data = validation_data[validation_rows]
        validation_targets = validation_targets.reshape(-1, 1)
        validation_targets = validation_targets[validation_rows]

    train_dataset = TensorDataset(torch.tensor(training_data, dtype=torch.float),
                                  torch.tensor(training_targets, dtype=torch.float))
    val_dataset = TensorDataset(torch.tensor(validation_data, dtype=torch.float),
                                torch.tensor(validation_targets, dtype=torch.float))

    return train_dataset, val_dataset, n_features
