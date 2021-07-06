import math
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import RMSprop
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from time import time

from modules.experiment_config import get_targets


def multi_task_losses(predicted_targets, targets, loss_function):
    losses = []

    for i in range(targets.shape[-1]):
        if len(targets.shape) == 3:
            predicted_target = predicted_targets[i][:, :, 0]
            target = targets[:, :, i]
        else:
            predicted_target = predicted_targets[i][:, 0]
            target = targets[:, i]
        losses.append(loss_function(predicted_target, target))

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
        losses.mean().backward()
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


def train_model(model_name, model, train_dataset, val_dataset, targets, epochs=10, batch_size=16, lr=0.001, seed=42):
    """
    Training the model using parameter inputs
    @param model_name: Parameter used for naming the checkpoint_dir
    @param model: model to be trained
    @param train_dataset: training dataset
    @param val_dataset: validation dataset
    @param targets:
    @param epochs: number of epochs to train
    @param batch_size: batch size
    @param lr: learning rate
    @param seed: random seed
    """
    torch.manual_seed(seed)
    target_range = range(len(targets))
    base_dir = './output/models'
    checkpoint_dir = f'{base_dir}/best_models'
    final_model_dir = f'{base_dir}/fully_trained_models'
    logs_dir = f'./output/logs'
    for directory in [checkpoint_dir, final_model_dir, logs_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    val_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    model = model.to(device)
    optimizer = RMSprop(model.parameters(), lr=lr, alpha=0.9)
    writer = SummaryWriter(log_dir=f'{logs_dir}/{model_name}_{time()}.log')

    best_val_loss = math.inf
    print(f'Training model {model_name} with {count_parameters(model)} parameters using {device}')
    start_time = time()
    for epoch in range(1, epochs + 1):
        train_losses = update(model, F.binary_cross_entropy, train_data_loader, optimizer, device)
        val_losses = evaluate(model, F.binary_cross_entropy, val_data_loader, device)
        val_loss = val_losses.mean()

        writer.add_scalar(f'Loss/train/Compound', train_losses.mean(), epoch)
        writer.add_scalar('Loss/val/Compound', val_loss, epoch)
        for i in target_range:
            writer.add_scalar(f'Loss/train/{targets[i]}', train_losses[i], epoch)
            writer.add_scalar(f'Loss/val/{targets[i]}', val_losses[i], epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model, f'{checkpoint_dir}/{model_name}.h5')
        torch.save(model, f'./{final_model_dir}/{model_name}.h5')
        print(f'\rEpoch {epoch:02d} out of {epochs} with loss {val_loss}', end=" ")

    print(f'\nTraining took {time() - start_time} seconds')

    if device != 'cpu':
        torch.cuda.empty_cache()
