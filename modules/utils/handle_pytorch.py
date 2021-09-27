"""Helper Functions for handling pytorch classes."""
import torch

import numpy as np
import xgboost as xgb

from torch.utils.data import TensorDataset

from modules.utils.handle_directories import load_pickle, get_pickle_file_path


def count_parameters(model: object):
    """
    Counts Parameters of a pytorch model.

    Parameters
    ----------
    model: object
        The model for which to count parameters

    Returns
    -------
    The number of parameters in a model

    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_weights_for_data(labels):
    """
    Calculate weights for labels.

    Parameters
    ----------
    labels: object
        Targets of a dataset

    Returns
    -------
    Sample weights for classes
    """
    n_samples = 0
    if len(labels.shape) == 3:
        num_labels = labels.sum(dim=1)
    else:
        num_labels = labels
    sample_weights = torch.zeros(num_labels.shape[0])
    for index in range(num_labels.shape[-1]):
        target = num_labels[:, index] > 0
        class_sample_count = torch.unique(target, return_counts=True)[1]
        n_samples = max(n_samples, max(class_sample_count))
        weight = 1.0 / class_sample_count.numpy()
        sample_weights += torch.tensor([weight[int(t)] for t in target])
    return sample_weights


def load_data_sets(data_path, target, n_percentage, reduce_dimensions=False):
    """
    Load data and convert it to Pytorch datasets.
    Optionally reduce the time dimension
    Parameters
    ----------
    data_path: str
        folder that contains the data files
    target: str
        variable to predict
    n_percentage: float
        percentage of full data to use
    reduce_dimensions: bool
        whether or not to reduce the time dimension

    Returns
    -------
    train data as Pytorch dataset as well as the number of features
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


def load_model(model_name, path='./output/models/best_models/'):
    """
    Loads a model.
    Determines which functions to use via model name
    Parameters
    ----------
    model_name: str
        file name of the model
    path: str
        folder in which model can be found
    Returns
    -------
    Loaded model
    """
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
