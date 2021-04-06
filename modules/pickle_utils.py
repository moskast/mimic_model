import pickle


def dump_pickle(variable, path):
    with open(path, 'wb') as file:
        pickle.dump(variable, file)


def load_pickle(path):
    with open(path, 'rb') as file:
        variable = pickle.load(file)
    return variable


def get_pickle_path(file_name, target, folder='.output/pickled_data_sets'):
    '''

    Returns: the path to the pickle folder with the filename structure
             {target}_{variable_name}.pickle in this case

    '''
    return f'{folder}/{target}_{file_name}.pickle'
