import copy
import math
import os
from time import time

import torch
import torch.nn.functional as F
from sklearn.model_selection import KFold
from torch.optim import RMSprop
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


def multi_task_losses(predicted_targets, targets, loss_function):
    losses = []
    # Vancomycin, MI, Sepsis
    weights = [1, 4, 15]
    #weights = [1, 1, 1]
    for i in range(targets.shape[-1]):
        if len(targets.shape) == 3:
            predicted_target = predicted_targets[i][:, :, 0]
            target = targets[:, :, i]
        else:
            predicted_target = predicted_targets[i][:, 0]
            target = targets[:, i]
        losses.append(loss_function(predicted_target, target)/weights[i])

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
    for input_data, targets in data_loader:
        input_data = input_data.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        predicted_targets = model(input_data)
        losses = multi_task_losses(predicted_targets, targets, loss_function)
        losses.sum(axis=0).backward()
        optimizer.step()
        train_loss.append(losses)
    train_losses = torch.stack(train_loss).mean(axis=0).detach()
    return train_losses


def evaluate(model, metric, data_loader, device='cpu'):
    model.eval()
    values = []
    with torch.no_grad():
        for input_data, targets in data_loader:
            input_data = input_data.to(device)
            targets = targets.to(device)
            output = model(input_data)
            value = multi_task_losses(output, targets, metric)
            values.append(value)
    metrics = torch.stack(values).mean(axis=0).detach()
    return metrics


def write_metric_list(writer, path, metrics, epoch, targets):
    writer.add_scalar(f'{path}/Compound', metrics.sum(), epoch)
    for i in range(len(targets)):
        writer.add_scalar(f'{path}{targets[i]}', metrics[i], epoch)


def train_model(model_name, og_model, dataset, targets, epochs=10, batch_size=128, lr=1e-3, k_folds=5, seed=0):
    """
    Training the model using parameter inputs
    @param model_name: Parameter used for naming the checkpoint_dir
    @param og_model: model to be trained
    @param dataset: training dataset
    @param targets:
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
    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
        writer = SummaryWriter(log_dir=f'{logs_dir}/{model_name}_{fold}_{time()}.log')
       # id_len = len(train_ids)
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)#[:id_len//25]
        val_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
        train_data_loader = DataLoader(dataset, sampler=train_subsampler, batch_size=batch_size)
        val_data_loader = DataLoader(dataset, sampler=val_subsampler, batch_size=batch_size)

        model = copy.deepcopy(og_model).to(device)
        optimizer = RMSprop(model.parameters(), lr=lr, alpha=0.9)

        best_val_loss = math.inf

        """train_losses = evaluate(model, F.binary_cross_entropy, train_data_loader, device)
        val_losses = evaluate(model, F.binary_cross_entropy, val_data_loader, device)

        write_metric_list(writer, writer_path_train, train_losses, 0, targets)
        write_metric_list(writer, writer_path_val, val_losses, 0, targets)"""

        print(f'\nTraining fold {fold + 1} out of {k_folds}')
        for epoch in range(1, epochs + 1):
            train_losses = update(model, F.binary_cross_entropy, train_data_loader, optimizer, device)
            val_losses = evaluate(model, F.binary_cross_entropy, val_data_loader, device)

            write_metric_list(writer, writer_path_train, train_losses, epoch, targets)
            write_metric_list(writer, writer_path_val, val_losses, epoch, targets)

            val_loss = val_losses.sum()
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model, f'{checkpoint_dir}/{model_name}_{fold}.h5')
            torch.save(model, f'./{final_model_dir}/{model_name}_{fold}.h5')
            print(f'\rEpoch {epoch:02d} out of {epochs} with loss {val_loss}', end=" ")

    print(f'\nTraining took {time() - start_time} seconds')

    if device != 'cpu':
        torch.cuda.empty_cache()
