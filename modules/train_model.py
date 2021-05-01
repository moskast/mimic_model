import math
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import RMSprop
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from modules.classes.model import ICU_LSTM, ICU_NN
from time import time

from modules.pickle_utils import load_pickle, get_pickle_path


def load_data(data_path, target, n_percentage, reduce_dimensions=False):
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


def train_LSTM(model_name, data_path, target, n_percentage, seed=42):
    train_dataset, val_dataset, n_features = load_data(data_path, target, n_percentage)
    model = ICU_LSTM(n_features)
    train_model(model_name, model, train_dataset, val_dataset, seed=seed)


def train_NN(model_name, data_path, target, n_percentage=1.0, seed=42):
    train_dataset, val_dataset, n_features = load_data(data_path, target, n_percentage, True)
    model = ICU_NN(n_features)
    train_model(model_name, model, train_dataset, val_dataset, seed=seed)


def return_loaded_model(model_name):
    return torch.load("./output/models/best_models/{0}.h5".format(model_name))


def update(model, loss_function, data_loader, optimizer, device='cpu'):
    model.train()
    train_loss = []
    for input_data, targets in data_loader:
        input_data = input_data.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        predicted_targets = model(input_data)
        loss = loss_function(predicted_targets, targets)
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())
    train_loss = np.mean(train_loss)
    return train_loss


def evaluate(model, metric, data_loader, device='cpu'):
    model.eval()
    values = []
    with torch.no_grad():
        for input_data, targets in data_loader:
            input_data = input_data.to(device)
            targets = targets.to(device)
            output = model(input_data)
            values.append(metric(output, targets).item())
    values = np.mean(values)
    return values


def train_model(model_name, model, train_dataset, val_dataset, epochs=13, batch_size=16, lr=0.001, seed=42):
    """

  Training the model using parameter inputs

  Args:
  ----
  model_name : Parameter used for naming the checkpoint_dir

  Return:
  -------
  Nonetype. Fits model only.

  """
    torch.manual_seed(seed)
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
    print(f'Training model {model_name} using {device}', end=" ")
    for epoch in range(1, epochs + 1):
        train_loss = update(model, F.binary_cross_entropy, train_data_loader, optimizer, device)
        writer.add_scalar('Loss/train', train_loss, epoch)

        val_loss = evaluate(model, F.binary_cross_entropy, val_data_loader, device)
        writer.add_scalar('Loss/val', val_loss, epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model, f'{checkpoint_dir}/{model_name}.h5')
        torch.save(model, f'./{final_model_dir}/{model_name}.h5')
        print(f'\rEpoch {epoch:02d} out of {epochs} with loss {val_loss}', end=" ")
    print()
    if device != 'cpu':
        torch.cuda.empty_cache()
