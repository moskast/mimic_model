import copy
import math
import os
from time import time

import numpy as np
import torch
from sklearn.model_selection import KFold
from torch.optim import RMSprop
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

import xgboost as xgb

from modules.models.multitask_loss_wrapper import MultiTaskLossWrapper


def multi_task_losses(predicted_targets, targets, loss_function):
    losses = []
    # Vancomycin, MI, Sepsis
    weights = [1, 4, 15]
    for i in range(targets.shape[-1]):
        if len(targets.shape) == 3:
            predicted_target = predicted_targets[i][:, :, 0]
            target = targets[:, :, i]
        else:
            predicted_target = predicted_targets[i][:, 0]
            target = targets[:, i]
        losses.append(loss_function(predicted_target, target) / weights[i])

    return torch.stack(losses)


def count_parameters(model):
    """
    Counts the number of parameters of target model
    @param model: model for which the parameters should be counted
    @return: number of parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def update(model, loss_function, data_loader, optimizer, device='cpu'):
    model.train()
    train_loss = []
    pos_ts = None
    neg_ts = None
    for input_data, targets in data_loader:
        t_sum = torch.sum(targets, dim=1) > 0
        if pos_ts is None:
            pos_ts = t_sum.sum(axis=0)
            neg_ts = targets.shape[0] - t_sum.sum(axis=0)
        else:
            pos_ts += t_sum.sum(axis=0)
            neg_ts += targets.shape[0] - t_sum.sum(axis=0)
        input_data = input_data.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        predicted_targets = model(input_data)
        """losses = multi_task_losses(predicted_targets, targets, loss_function)
        losses.sum(axis=0).backward()"""
        combined_loss, losses = loss_function(predicted_targets, targets)
        combined_loss.backward()
        optimizer.step()
        train_loss.append(losses)
    #print(f'\nLabel difference of: {neg_ts / pos_ts} - {pos_ts=} {neg_ts=}')
    train_losses = torch.stack(train_loss).mean(axis=0).detach()
    return train_losses


def evaluate(model, metric, data_loader, device='cpu'):
    model.eval()
    values = []
    with torch.no_grad():
        for input_data, targets in data_loader:
            input_data = input_data.to(device)
            targets = targets.to(device)
            predicted_targets = model(input_data)
            # value = multi_task_losses(output, targets, metric)
            _, value = metric(predicted_targets, targets)
            values.append(value)
    metrics = torch.stack(values).mean(axis=0).detach()
    return metrics


def write_metric_list(writer, path, metrics, epoch, targets):
    if len(targets) >= 2:
        writer.add_scalar(f'{path}/Compound', metrics.sum(), epoch)
    for i in range(len(targets)):
        writer.add_scalar(f'{path}{targets[i]}', metrics[i], epoch)


def write_model_weights(writer, model, epoch):
    for name, weight in model.named_parameters():
        name, label = name.split('.', 1)
        writer.add_histogram(f'{name}/{label}', weight, epoch)
        if epoch != 0 and weight.grad is not None:
            writer.add_histogram(f'{name}/{label}_grad', weight.grad, epoch)


def get_weights_for_data(labels):
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
        weight = 1. / class_sample_count.numpy()
        sample_weights += torch.tensor([weight[int(t)] for t in target])
    return sample_weights


def train_model(model_name, og_model, dataset, target_names, oversample=False,
                epochs=10, batch_size=128, lr=1e-3, k_folds=5, seed=0):
    """
    Training the model using parameter inputs
    @param oversample:
    @param model_name: Parameter used for naming the checkpoint_dir
    @param og_model: model to be trained
    @param dataset: training dataset
    @param target_names:
    @param epochs: number of epochs to train
    @param batch_size: batch size
    @param lr: learning rate
    @param k_folds:
    @param seed: random seed
    """
    torch.manual_seed(seed)
    base_dir = './output/models'
    checkpoint_dir = f'{base_dir}/best_models'
    final_model_dir = f'{base_dir}/fully_trained_models'
    logs_dir = f'./output/logs'
    writer_path_train = 'Loss/train/'
    writer_path_val = 'Loss/val/'
    for directory in [checkpoint_dir, final_model_dir, logs_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=seed)

    start_time = time()
    print(f'Training model {model_name} with {count_parameters(og_model)} parameters using {device}')
    for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):
        writer = SummaryWriter(log_dir=f'{logs_dir}/{model_name}_{fold}_{time()}.log')

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

        write_metric_list(writer, writer_path_train, train_losses, 0, target_names)
        write_metric_list(writer, writer_path_val, val_losses, 0, target_names)
        write_model_weights(writer, model, 0)

        print(f'\nTraining fold {fold + 1} out of {k_folds}')
        for epoch in range(1, epochs + 1):
            train_losses = update(model, loss_function, train_data_loader, optimizer, device)
            val_losses = evaluate(model, loss_function, val_data_loader, device)

            val_loss = val_losses.sum()
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model, f'{checkpoint_dir}/{model_name}_{fold}.h5')
            torch.save(model, f'./{final_model_dir}/{model_name}_{fold}.h5')
            print(f'\rEpoch {epoch:02d} out of {epochs} with loss {val_loss}', end=" ")

            write_metric_list(writer, writer_path_train, train_losses, epoch, target_names)
            write_metric_list(writer, writer_path_val, val_losses, epoch, target_names)
            write_model_weights(writer, model, epoch)

            scheduler.step()

        break

    print(f'\nTraining took {time() - start_time} seconds')

    if device != 'cpu':
        torch.cuda.empty_cache()


def train_xgb(model_name, dataset, oversample=False,
              esr=50, nbr=100, lr=0.15, npt=100, k_folds=5, seed=0):
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

    base_dir = './output/models'
    checkpoint_dir = f'{base_dir}/best_models'
    final_model_dir = f'{base_dir}/fully_trained_models'
    for directory in [checkpoint_dir, final_model_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    start_time = time()
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
    print(f'Training model {model_name}')
    for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):
        print(f'\nTraining fold {fold + 1} out of {k_folds}')
        train_data, train_labels = dataset[train_ids]
        val_data, val_labels = dataset[val_ids]

        if oversample:
            sample_weights = get_weights_for_data(train_labels)
        else:
            sample_weights = np.ones(train_labels.shape[0])

        d_train = xgb.DMatrix(train_data.numpy(), train_labels.numpy(), weight=sample_weights)
        d_val = xgb.DMatrix(val_data.numpy(), val_labels.numpy())
        evallist = [(d_train, 'train'), (d_val, 'eval')]
        bst = xgb.train(params, d_train, evals=evallist, early_stopping_rounds=esr, verbose_eval=50,
                        num_boost_round=nbr)
        print(f'{bst.best_iteration} - {bst.best_score=}')
        bst.save_model(f'{final_model_dir}/{model_name}_{fold}.model')
        bst.save_model(f'{checkpoint_dir}/{model_name}_{fold}.model')
        print("Saved model")
        break

    print(f'\nTraining took {time() - start_time} seconds')
