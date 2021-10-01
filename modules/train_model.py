import copy
import math
import os
from time import time

import torch
import xgboost as xgb
from sklearn.model_selection import KFold
from torch.optim import RMSprop
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader, Subset

from modules.classes.tensorboard_writer import TensorboardWriter
from modules.utils.handle_pytorch import count_parameters, get_weights_for_data
from modules.utils.handle_directories import get_train_folders
from modules.models.multitask_loss_wrapper import MultiTaskLossWrapper


def update(model, loss_function, data_loader, optimizer, device='cpu'):
    """
    Train model for one epoch.
    Parameters
    ----------
    model: object
        Model which gets trained
    loss_function: object
        Loss function for calculating gradients
    data_loader: object
        Contains training data
    optimizer: object
        Algorithm for weight updates
    device: str
        Determines where to do the computations
    Returns
    -------
    A 1-D array of train losses
    """
    model.train()
    train_loss = []
    for input_data, targets in data_loader:
        input_data = input_data.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        predicted_targets = model(input_data)
        combined_loss, losses = loss_function(predicted_targets, targets)
        combined_loss.backward()
        optimizer.step()
        train_loss.append(losses)
    train_losses = torch.stack(train_loss).mean(axis=0).detach()
    return train_losses


def evaluate(model, metric, data_loader, device='cpu'):
    """
    Evaluate model on all data.
    Parameters
    ----------
    model: object
        Model which gets evaluated
    metric: object
        Function which determines the difference between predictions and labels
    data_loader: object
        Contains evaluation data
    device: str
        Determines where to do the computations
    Returns
    -------
    A 1-D array of losses
    """
    model.eval()
    values = []
    with torch.no_grad():
        for input_data, targets in data_loader:
            input_data = input_data.to(device)
            targets = targets.to(device)
            predicted_targets = model(input_data)
            _, value = metric(predicted_targets, targets)
            values.append(value)
    metrics = torch.stack(values).mean(axis=0).detach()
    return metrics


def train_model(model_name, og_model, dataset, target_names, oversample=False,
                epochs=10, batch_size=256, lr=1e-3, k_folds=3, seed=0):
    """
    Main training function for Pytorch models.
    Trains the given model for the given amount of epochs.
    After each Epoch the model, its weights
    as well as the train and validation performance are saved.
    Parameters
    ----------
    model_name: str
        Name of the model, used for the filename of the saved model
    og_model: object
        Model to be trained
    dataset: object
        training dataset
    target_names: list[str]
        Targets for which the model is trained
    oversample: bool
        Whether or not to oversample
    epochs: int
        Number of times the model sees every data point
    batch_size: int
        Number of data points per batch
    lr: float
        Learning Rate, defines the size of the updates
    k_folds: int
        Number of folds to create for Cross Validation
    seed: int
        random seed for reproducibility
    """
    torch.manual_seed(seed)
    best_model_dir, final_model_dir, logs_dir = get_train_folders()
    writer_path_train = 'Loss/train/'
    writer_path_val = 'Loss/val/'
    for directory in [best_model_dir, final_model_dir, logs_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    k_fold = KFold(n_splits=k_folds, shuffle=True, random_state=seed)

    start_time = time()
    print(f'Training model {model_name} with {count_parameters(og_model)} parameters using {device}')
    for fold, (train_ids, val_ids) in enumerate(k_fold.split(dataset)):
        writer = TensorboardWriter(f'{logs_dir}/{model_name}_{fold}_{time()}.log', target_names)

        targets = dataset[train_ids][1]
        if oversample:
            sample_weights = get_weights_for_data(targets)
            sampler = torch.utils.data.WeightedRandomSampler(sample_weights, len(sample_weights))
        else:
            sampler = None

        train_data_loader = DataLoader(Subset(dataset, train_ids), sampler=sampler, batch_size=batch_size)
        val_data_loader = DataLoader(Subset(dataset, val_ids), batch_size=batch_size)

        model = copy.deepcopy(og_model).to(device)
        loss_function = MultiTaskLossWrapper(len(target_names)).to(device)

        optimizer = RMSprop(model.parameters(), lr=lr, alpha=0.9)
        scheduler = ExponentialLR(optimizer, gamma=0.9)

        best_val_loss = math.inf

        train_losses = evaluate(model, loss_function, train_data_loader, device)
        val_losses = evaluate(model, loss_function, val_data_loader, device)

        writer.save_metric_list(writer_path_train, train_losses, 0)
        writer.save_metric_list(writer_path_val, val_losses, 0)
        writer.save_model_weights(model, 0)

        print(f'\nTraining fold {fold + 1} out of {k_folds}')
        for epoch in range(1, epochs + 1):
            train_losses = update(model, loss_function, train_data_loader, optimizer, device)
            val_losses = evaluate(model, loss_function, val_data_loader, device)

            val_loss = val_losses.sum()
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model, f'{best_model_dir}{model_name}_{fold}.h5')
            torch.save(model, f'./{final_model_dir}{model_name}_{fold}.h5')
            print(f'\rEpoch {epoch:02d} out of {epochs} with loss {val_loss}', end=" ")

            writer.save_metric_list(writer_path_train, train_losses, epoch)
            writer.save_metric_list(writer_path_val, val_losses, epoch)
            writer.save_model_weights(model, epoch)

            scheduler.step()

    print(f'\nTraining took {time() - start_time} seconds')

    if device != 'cpu':
        torch.cuda.empty_cache()


def train_xgb(model_name, dataset, oversample=False,
              esr=50, nbr=50, lr=0.15, npt=100, k_folds=3, seed=0):
    """
    Main training function for Pytorch models.
    Trains the given model for the given amount of epochs.
    Parameters
    ----------
    model_name: str
        Name of the model, used for the filename of the saved model
    dataset: object
        training dataset
    oversample: bool
        Whether or not to oversample
    esr: int
        XGBoost parameter
    nbr: int
        XGBoost parameter
    lr: float
        Learning Rate, defines the size of the updates
    npt: int
        XGBoost parameter
    k_folds: int
        Number of folds to create for Cross Validation
    seed: int
        random seed for reproducibility
    """
    params = {
        'learning_rate': lr,
        'gamma': 0,
        'max_depth': 5,
        'min_child_weight': 5,
        'num_parallel_tree': npt,
        'subsample': 0.8,
        'colsample_bynode': 0.8,
        'tree_method': 'gpu_hist',
        'objective': 'binary:logistic',
        'eval_metric': ['error', 'auc'],
        'seed': seed
    }

    checkpoint_dir, final_model_dir, logs_dir = get_train_folders()
    for directory in [checkpoint_dir, final_model_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    start_time = time()
    k_fold = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
    print(f'Training model {model_name}')
    for fold, (train_ids, val_ids) in enumerate(k_fold.split(dataset)):
        print(f'\nTraining fold {fold + 1} out of {k_folds}')
        train_data, train_labels = dataset[train_ids]
        val_data, val_labels = dataset[val_ids]

        if oversample:
            sample_weights = get_weights_for_data(train_labels).numpy()
        else:
            sample_weights = None

        d_train = xgb.DMatrix(train_data.numpy(), train_labels.numpy(), weight=sample_weights)
        d_val = xgb.DMatrix(val_data.numpy(), val_labels.numpy())
        eval_list = [(d_train, 'train'), (d_val, 'eval')]
        bst = xgb.train(params, d_train, evals=eval_list, early_stopping_rounds=esr, verbose_eval=50,
                        num_boost_round=nbr)
        print(f'{bst.best_iteration} - {bst.best_score=}')
        bst.save_model(f'{final_model_dir}/{model_name}_{fold}.model')
        bst.save_model(f'{checkpoint_dir}/{model_name}_{fold}.model')
        print("Saved model")

    print(f'\nTraining took {time() - start_time} seconds')
