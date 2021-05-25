import pickle


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


def get_pickle_path(file_name, target, folder='.output/pickled_data_sets'):
    """
    Returns a path to a saved file
    @param file_name: name of the saved file
    @param target: name of target column
    @param folder: name of the folder file is saved in
    @return: the path to the pickle folder with the filename structure {target}_{variable_name}.pickle in this case
    """
    return f'{folder}/{target}_{file_name}.pickle'
