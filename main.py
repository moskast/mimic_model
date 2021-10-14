import time

from modules.config import AppConfig
from modules.classes.mimic_parser import MimicParser
from modules.classes.mimic_pre_processor import MimicPreProcessor
from modules.utils.handle_directories import get_pickle_folder
from modules.models.attention_models import AttentionLSTM
from modules.models.comparison_models import ComparisonLSTM, ComparisonFNN, ComparisonLogisticRegression
from modules.models.hopfield_models import HopfieldLayerModel, HopfieldPoolingModel, HopfieldLookupModel, HopfieldLSTM
from modules.train_model import train_model, train_xgb
from modules.utils.handle_pytorch import load_data_sets


def train_models(mimic_version, data_path, n_time_steps, random_seed, targets):
    """
    Training loop for training models with targets and percentages
    Parameters
    ----------
    mimic_version: int
        which mimic version to use 3 or 4
    data_path: str
        path to data for the experiments
    n_time_steps: int
        number of time step for one sample
    random_seed: list[int]
        seed for setting random functions
    targets
    """
    start_time = time.time()
    print(f'{data_path=}')
    n_targets = len(targets)
    print(f'\nTarget: {targets}')
    for p in AppConfig.percentages:
        print(f'Percentage: {p}')
        train_dataset, n_features = load_data_sets(data_path, targets, p)
        common_model_id = f'_{mimic_version}_{targets}_{n_time_steps}_{random_seed}'
        train_dataset_reduced, n_features_reduced = load_data_sets(data_path, targets, p, reduce_dimensions=True)

        if len(targets) == 1:  # If not Multitasking
            model_id = 'xgb' + common_model_id
            train_xgb(model_id, train_dataset_reduced, seed=random_seed,
                      oversample=AppConfig.oversample, k_folds=AppConfig.k_folds)
            model_id = 'random_forest_xgb' + common_model_id
            train_xgb(model_id, train_dataset_reduced, nbr=1, lr=1, npt=100, seed=random_seed,
                      oversample=AppConfig.oversample, k_folds=AppConfig.k_folds)

        models = [
            ('comparison_LR', ComparisonLogisticRegression(n_features_reduced, num_targets=n_targets)),
            ('comparison_FNN', ComparisonFNN(n_features_reduced, num_targets=n_targets)),
            ('comparison_LSTM', ComparisonLSTM(n_features, num_targets=n_targets)),
            ('partial_attention_LSTM', AttentionLSTM(n_features, full_attention=False, num_targets=n_targets)),
            ('full_attention_LSTM', AttentionLSTM(n_features, full_attention=True, num_targets=n_targets)),
            ('partial_hopfield_LSTM', HopfieldLSTM(n_features, full_attention=False, num_targets=n_targets)),
            ('full_hopfield_LSTM', HopfieldLSTM(n_features, full_attention=True, num_targets=n_targets)),
            ('hopfield_layer', HopfieldLayerModel(n_features, num_targets=n_targets)),
            ('hopfield_pooling', HopfieldPoolingModel(n_features, num_targets=n_targets)),
            ('hopfield_lookup', HopfieldLookupModel(n_features, int(len(train_dataset) / 10000), num_targets=n_targets))
        ]

        for model_name, model in models:
            model_id = model_name + common_model_id
            if model_name == 'comparison_FNN' or model_name == 'comparison_LR':
                train_model(model_id, model, train_dataset_reduced, targets, seed=random_seed,
                            oversample=AppConfig.oversample, k_folds=AppConfig.k_folds)
            else:
                train_model(model_id, model, train_dataset, targets,
                            seed=random_seed, oversample=AppConfig.oversample, k_folds=AppConfig.k_folds)

        print(f'\rFinished training on {p * 100}% of data')
    print(f'\rFinished training on {random_seed=}')
    end_time = time.time()
    print(f'{end_time - start_time} seconds needed for training')


def main(parse_mimic, pre_process_data, create_models):
    """
    Main loop that process mimic db, preprocess data and trains models
    Parameters
    ----------
    parse_mimic: bool
        whether to parse the mimic database
    pre_process_data: bool
        whether to preprocess the parsed the mimic database
    create_models: bool
        whether to train the models
    """
    mimic_version = AppConfig.mimic_version
    window_size = AppConfig.window_size
    random_seed = AppConfig.random_seed
    stat = 'with_stat' if AppConfig.create_statistics else 'no_stat'
    print('Start Program')
    print(f'Mimic Version {mimic_version}')
    original_mimic_folder = f'./data/mimic_{mimic_version}_database'
    parsed_mimic_folder = f'mapped_elements_ws_{window_size}_{stat}'
    file_name = 'CHARTEVENTS'
    id_col = 'ITEMID'
    label_col = 'LABEL'

    mimic_parser = MimicParser(original_mimic_folder, parsed_mimic_folder, file_name, id_col, label_col, mimic_version)

    # Parse Mimic
    if parse_mimic:
        print('Parse Mimic Data')
        mimic_parser.perform_full_parsing(window_size=window_size, create_statistics=AppConfig.create_statistics)

    n_time_steps = int((24 // window_size) * 14)
    pickled_data_path = get_pickle_folder(mimic_version, n_time_steps, AppConfig.create_statistics,
                                          AppConfig.balance_data, random_seed)

    targets = AppConfig.targets
    # Preprocess Mimic
    if pre_process_data:
        print('Preprocess Data')
        if mimic_version == 3:
            parsed_mimic_filepath = mimic_parser.an_path + '.csv'
        else:
            parsed_mimic_filepath = mimic_parser.aii_path + '.csv'

        mimic_pp = MimicPreProcessor(parsed_mimic_filepath, random_seed=random_seed)

        print(f'Creating Datasets for {targets}')
        mimic_pp.apply_pipeline(targets, n_time_steps, pickled_data_path, balance_set=AppConfig.balance_data)
        for target in targets:
            mimic_pp.apply_pipeline([target], n_time_steps, pickled_data_path, balance_set=AppConfig.balance_data)
        print(f'Created Datasets for {targets}\n')

    if create_models:
        train_models(mimic_version, pickled_data_path, n_time_steps, random_seed, targets)
        if AppConfig.train_single_targets and len(targets) > 1:
            for target in targets:
                train_models(mimic_version, pickled_data_path, n_time_steps, random_seed, [target])


if __name__ == "__main__":
    parse = False
    pre_process = False
    train = True

    '''for bd in [True, False]:
        AppConfig.balance_data = bd
        main(parse, pre_process, train)'''

    for bd, os in [(True, False), (False, False), (False, True)]:
        AppConfig.balance_data = bd
        AppConfig.oversample = os
        print(f'{AppConfig.oversample=} - {AppConfig.balance_data=}')
        main(parse, pre_process, train)
