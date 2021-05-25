from modules.classes.mimic_parser import MimicParser
from modules.classes.mimic_pre_processor import MimicPreProcessor
from modules.train_model import train_LSTM, train_NN, train_LSTM_Attention


def get_targets():
    """
    @return: The targets for the experiments
    """
    return ['VANCOMYCIN']
    return ['MI', 'SEPSIS', 'VANCOMYCIN']


def get_percentages():
    """
    @return: Percentages of training data to be used for the experiments
    """
    return [1.0]


def get_pickle_folder(mimic_version, n_time_steps):
    """
    @return: path to data for the experiments
    """
    return f'./data/pickled_data_sets/mimic_{mimic_version}/{n_time_steps}_ts'


def train_models(targets, percentages, mimic_version, data_path, n_time_steps):
    """
    Training loop for training models with targets and percentages
    @param targets: targets for the experiments
    @param percentages: percentages of training data to be used for the experiments
    @param mimic_version: which mimic version to use 3 or 4
    @param data_path: path to data for the experiments
    @param n_time_steps: number of time step for one sample
    """
    for target in targets:
        print(f'\nTraining {target}')
        for percentage in percentages:
            p = int(percentage * 100)
            for seed in range(5):
                print('Training NN')
                model_name = f'mimic_NN_{mimic_version}_{target}_{n_time_steps}_{seed}'
                train_NN(model_name=model_name, target=target, n_percentage=percentage, data_path=data_path, seed=seed)
                print('Training LSTM')
                model_name = f'mimic_LSTM_{mimic_version}_{target}_{n_time_steps}_{seed}'
                train_LSTM(model_name=model_name, target=target, n_percentage=percentage, data_path=data_path, seed=seed)
                print('Training LSTM Attention')
                model_name = f'mimic_LSTM_Attention_{mimic_version}_{target}_{n_time_steps}_{seed}'
                train_LSTM_Attention(model_name=model_name, target=target, n_percentage=percentage, data_path=data_path, seed=seed)
                print(f'\rFinished training on {seed=}')
            print(f'\rFinished training on {percentage * 100}% of data')


def main(parse_mimic, pre_process_data, create_models, mimic_version, window_size):
    """
    Main loop that process mimic db, preprocess data and trains models
    @param parse_mimic: whether to parse the mimic database
    @param pre_process_data: whether to preprocess the parsed the mimic database
    @param create_models: whether to train the models
    @param mimic_version: which mimic version to use 3 or 4
    @param window_size: number of hours for one time step
    """
    print('Start Program')
    print(f'Mimic Version {mimic_version}')
    original_mimic_folder = f'./data/mimic_{mimic_version}_database'
    parsed_mimic_folder = f'mapped_elements_ws_{window_size}'
    file_name = 'CHARTEVENTS'
    id_col = 'ITEMID'
    label_col = 'LABEL'

    mimic_parser = MimicParser(original_mimic_folder, parsed_mimic_folder, file_name, id_col, label_col, mimic_version,
                               window_size)

    # Parse Mimic
    if parse_mimic:
        print('Parse Mimic Data')
        mimic_parser.perform_full_parsing()

    n_time_steps = int((24//window_size) * 14)
    pickled_data_path = get_pickle_folder(mimic_version, n_time_steps)

    # Preprocess Mimic
    if pre_process_data:
        print('Preprocess Data')
        if mimic_version == 3:
            parsed_mimic_filepath = mimic_parser.an_path + '.csv'
        else:
            parsed_mimic_filepath = mimic_parser.aii_path + '.csv'

        targets = get_targets()
        mimic_pre_processor = MimicPreProcessor(parsed_mimic_filepath)

        for target in targets:
            print(f'Creating Datasets for {target}')
            mimic_pre_processor.pre_process_and_save_files(target, n_time_steps, pickled_data_path)
            print(f'Created Datasets for {target}\n')

    if create_models:
        train_models(get_targets(), get_percentages(), mimic_version, pickled_data_path, n_time_steps)


if __name__ == "__main__":
    parse = False
    pre_process = False
    train = True
    mimic_v = 4
    window_s = 24
    main(parse, pre_process, train, mimic_v, window_s)
