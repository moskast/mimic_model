import pickle

from modules.config import AppConfig


def dump_pickle(variable, path):
    """
    Save a file via pickle
    Parameters
    ----------
    variable: object
        file to save
    path: str
        path of the saved file
    """
    with open(path, 'wb') as file:
        pickle.dump(variable, file)


def load_pickle(path):
    """
    Load a saved pickle file
    Parameters
    ----------
    path: str
        path of the saved file
    Returns
    -------
    loaded pickle file
    """
    with open(path, 'rb') as file:
        variable = pickle.load(file)
    return variable


def get_output_directory(create_statistics=None, undersample=None, oversample=None):
    """
    Determines the output directory given the balance of the dataset as well as columns.
    Parameters
    ----------
    create_statistics: bool
        Whether the std, min and max columns have been created
    undersample: bool
        Whether the data has been undersampled
    oversample: bool
        Whether the data has been oversampled

    Returns
    -------
    Output directory
    """
    if create_statistics is None:
        create_statistics = AppConfig.create_statistics
    if undersample is None:
        undersample = AppConfig.balance_data
    if oversample is None:
        oversample = AppConfig.oversample
    stat = 'st' if create_statistics else 'ns'
    bal = 'us' if undersample else 'ub'
    bal = 'os' if oversample else bal
    return f'./output/{stat}_{bal}/'


def get_train_folders(create_statistics=None, undersample=None, oversample=None):
    """
    Determines the output folders for the training routine.
    Parameters
    ----------
    create_statistics: bool
        Whether the std, min and max columns have been created
    undersample: bool
        Whether the data has been undersampled
    oversample: bool
        Whether the data has been oversampled

    Returns
    -------
    Checkpoint, Final model and Log directory
    """
    base_dir = get_output_directory(create_statistics, undersample, oversample)
    model_dir = f'{base_dir}models/'
    checkpoint_dir = f'{model_dir}best_models/'
    final_model_dir = f'{model_dir}fully_trained_models/'
    logs_dir = f'{base_dir}logs'
    return checkpoint_dir, final_model_dir, logs_dir


def get_figure_dir(create_statistics=None, undersample=None, oversample=None):
    """
    Determines the figure directory.
    Parameters
    ----------
    create_statistics: bool
        Whether the std, min and max columns have been created
    undersample: bool
        Whether the data has been undersampled
    oversample: bool
        Whether the data has been oversampled

    Returns
    -------
    Figure directory
    """
    base_dir = get_output_directory(create_statistics, undersample, oversample)
    return base_dir + 'figures'


def get_pickle_folder(mimic_version, n_time_steps,
                      created_statistics=AppConfig.create_statistics, balanced=AppConfig.balance_data,
                      seed=AppConfig.random_seed):
    """
    Determines folder for the pickled data
    Parameters
    ----------
    mimic_version: int
        Mimic Version
    n_time_steps: int
        Number of time steps
    created_statistics: bool
        Whether the std, min and max columns have been created
    balanced: bool
        Whether the data has been undersampled
    seed: int
        Random seed used for data processing

    Returns
    -------
    path to data for the experiments
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

    Parameters
    ----------
    file_name: str
        name of the saved file
    target: str
        name of target column
    folder: str
        name of the folder file is saved in

    Returns
    -------
    path to the pickle folder with the filename structure {target}_{variable_name}.pickle in this case
    """
    return f'{folder}/{target}_{file_name}.pickle'
