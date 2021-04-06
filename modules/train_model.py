import math
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import RMSprop
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from modules.classes.model import ICU_LSTM
from time import time

from modules.pickle_utils import load_pickle, get_pickle_path


def return_loaded_model(model_name):
    return torch.load("./output/saved_models/best_models/{0}.h5".format(model_name))


def update(model, loss_function, data_loader, optimizer, device='cpu'):
    model.train()
    train_loss = []
    for input_data, targets in data_loader:
        input_data = input_data.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        predicted_targets, _ = model(input_data)
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
            output, _ = model(input_data)
            values.append(metric(output, targets).item())
    values = np.mean(values)
    return values


def train_model(model_name, target='MI', n_percentage=1.0,
                epochs=10, batch_size=32, lr=0.001, data_path='pickled_data_sets', seed=42):
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
    base_dir = './output/saved_models'
    checkpoint_dir = f'{base_dir}/best_models'
    final_model_dir = f'{base_dir}/fully_trained_models'
    logs_dir = f'{base_dir}/logs'
    for directory in [checkpoint_dir, final_model_dir, logs_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    training_data = load_pickle(get_pickle_path('train_data', target, data_path))
    training_targets = load_pickle(get_pickle_path('train_targets', target, data_path))
    validation_data = load_pickle(get_pickle_path('validation_data', target, data_path))
    validation_targets = load_pickle(get_pickle_path('validation_target', target, data_path))
    number_feature_cols = training_data.shape[2]

    # N_Samples x Seq_Length x N_Features
    training_data = training_data[0:int(n_percentage * training_data.shape[0])]  # Subsample if necessary
    training_targets = training_targets[0:int(n_percentage * training_targets.shape[0])]

    train_dataset = TensorDataset(torch.tensor(training_data, dtype=torch.float),
                                  torch.tensor(training_targets, dtype=torch.float))
    val_dataset = TensorDataset(torch.tensor(validation_data, dtype=torch.float),
                                torch.tensor(validation_targets, dtype=torch.float))
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    val_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    model = ICU_LSTM(number_feature_cols).to(device)
    optimizer = RMSprop(model.parameters(), lr=lr, alpha=0.9)
    writer = SummaryWriter(log_dir=f'{logs_dir}/{model_name}_{time()}.log')

    best_val_loss = math.inf
    print(f'Training Models using {device}', end=" ")
    for epoch in range(1, epochs + 1):
        print(f'\rEpoch {epoch:02d} out of {epochs} with loss {best_val_loss}', end=" ")

        train_loss = update(model, F.binary_cross_entropy, train_data_loader, optimizer, device)
        writer.add_scalar('Loss/train', train_loss, epoch)

        val_loss = evaluate(model, F.binary_cross_entropy, val_data_loader, device)
        writer.add_scalar('Loss/val', val_loss, epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model, f'{checkpoint_dir}/{model_name}.h5')

    torch.save(model, f'./{final_model_dir}/{model_name}.h5')

    if device != 'cpu':
        torch.cuda.empty_cache()


def main(targets, percentages, seed=42):
    """

    Args:
        seed: random seed for pytorch
        create_data: If the data should be create, False by default

    Returns:

    """

    epochs = 13
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(seed)
    print(f'Training Models using {device}')
    for target in targets:
        print(f'\nTraining {target}')
        for percentage in percentages:
            p = int(percentage * 100)
            model_name = f'mimic_{target}_{p}_percent'
            train_model(model_name=model_name, epochs=epochs, device=device,
                        target=target, n_percentage=percentage)

            torch.cuda.empty_cache()
            print(f'\rFinished training on {percentage * 100}% of data')
    print("Program finished")
