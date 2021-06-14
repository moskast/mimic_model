import time
from modules.classes.mimic_parser import MimicParser
from modules.classes.mimic_pre_processor import MimicPreProcessor
from modules.experiment_config import get_targets, get_percentages, get_seeds
from modules.load_data import load_data_sets, get_pickle_folder
from modules.models.attention_models import AttentionLSTM
from modules.models.comparison_models import ComparisonLSTM, ComparisonFNN
from modules.models.hopfield_models import HopfieldLayerModel, HopfieldPoolingModel, HopfieldLookupModel
from modules.train_model import train_model


def train_models(mimic_version, data_path, n_time_steps, train_comparison=False):
    """
    Training loop for training models with targets and percentages
    @param mimic_version: which mimic version to use 3 or 4
    @param data_path: path to data for the experiments
    @param n_time_steps: number of time step for one sample
    @param train_comparison: whether to train for NN-LSTM comparison or benchmark experiment
    """
    start_time = time.time()
    for target in get_targets():
        print(f'\nTarget: {target}')
        for p in get_percentages():
            print(f'Percentage: {p}')
            train_dataset, val_dataset, n_features = load_data_sets(data_path, target, p)
            if train_comparison:
                train_dataset_reduced, val_dataset_reduced, n_features_reduced = load_data_sets(data_path, target, p,
                                                                                                True)
            for random_seed in get_seeds():
                common_model_id = f'_{mimic_version}_{target}_{n_time_steps}_{random_seed}'
                if train_comparison:
                    model_id = 'comparison_LSTM' + common_model_id
                    model = ComparisonLSTM(n_features)
                    train_model(model_id, model, train_dataset, val_dataset, seed=random_seed)
                    print('Training NN')
                    model_id = 'comparison_FNN' + common_model_id
                    model = ComparisonFNN(n_features_reduced)
                    train_model(model_id, model, train_dataset_reduced, val_dataset_reduced, seed=random_seed)
                    print('Training LSTM')

                else:
                    models = [('partial_attention_LSTM', AttentionLSTM(n_features, full_attention=False)),
                              ('full_attention_LSTM', AttentionLSTM(n_features, full_attention=True)),
                              ('hopfield_layer', HopfieldLayerModel(n_features)),
                              ('hopfield_pooling', HopfieldPoolingModel(n_features)),
                              ('hopfield_lookup', HopfieldLookupModel(n_features, int(len(train_dataset) / 1000)))]
                    for model_name, model in models:
                        model_id = model_name + common_model_id
                        train_model(model_id, model, train_dataset, val_dataset, seed=random_seed)

                print(f'\rFinished training on {random_seed=}')
            print(f'\rFinished training on {p * 100}% of data')
    end_time = time.time()
    print(f'{end_time - start_time} seconds needed for training')


def main(parse_mimic, pre_process_data, create_models, mimic_version, window_size, random_seed=42):
    """
    Main loop that process mimic db, preprocess data and trains models
    @param random_seed: random seed
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

    n_time_steps = int((24 // window_size) * 14)
    pickled_data_path = get_pickle_folder(mimic_version, n_time_steps, random_seed)

    # Preprocess Mimic
    if pre_process_data:
        print('Preprocess Data')
        if mimic_version == 3:
            parsed_mimic_filepath = mimic_parser.an_path + '.csv'
        else:
            parsed_mimic_filepath = mimic_parser.aii_path + '.csv'

        targets = get_targets()
        mimic_pre_processor = MimicPreProcessor(parsed_mimic_filepath, random_seed=random_seed)

        for target in targets:
            print(f'Creating Datasets for {target}')
            mimic_pre_processor.pre_process_and_save_files(target, n_time_steps, pickled_data_path)
            print(f'Created Datasets for {target}\n')

    if create_models:
        train_models(mimic_version, pickled_data_path, n_time_steps)


if __name__ == "__main__":
    parse = False
    pre_process = True
    train = False
    mimic_v = 4
    window_s = 24
    main(parse, pre_process, train, mimic_v, window_s)
