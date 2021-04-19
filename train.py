from modules.classes.mimic_parser import MimicParser
from modules.classes.mimic_pre_processor import MimicPreProcessor
from modules.train_model import train_model


def get_targets():
    return ['MI', 'SEPSIS', 'VANCOMYCIN']


def get_percentages():
    return [1.0]


def get_pickle_folder(mimic_version):
    return f'./data/pickled_data_sets/mimic_{mimic_version}'


def train_models(targets, percentages, mimic_version):
    data_path = get_pickle_folder(mimic_version)
    for target in targets:
        print(f'\nTraining {target}')
        for percentage in percentages:
            p = int(percentage * 100)
            for seed in range(3):
                model_name = f'mimic_{mimic_version}_{target}_{p}_percent_{seed}'
                train_model(model_name=model_name, target=target, n_percentage=percentage, data_path=data_path,
                            seed=seed)
                print(f'\rFinished training on {seed=}')
            print(f'\rFinished training on {percentage * 100}% of data')


def main(parse_mimic, pre_process_data, create_models, mimic_version, window_size):
    print('Start Program')
    print(f'Mimic Version {mimic_version}')
    mimic_folder_path = f'./data/mimic_{mimic_version}_database'
    output_folder = f'mapped_elements_{window_size}'
    file_name = 'CHARTEVENTS'
    id_col = 'ITEMID'
    label_col = 'LABEL'

    mimic_parser = MimicParser(mimic_folder_path, output_folder, file_name, id_col, label_col, mimic_version,
                               window_size)

    # Parse Mimic
    if parse_mimic:
        print('Parse Mimic Data')
        mimic_parser.perform_full_parsing()

    # Preprocess Mimic
    if pre_process_data:
        print('Preprocess Data')
        if mimic_version == 3:
            output_filepath = mimic_parser.an_path + '.csv'
        else:
            output_filepath = mimic_parser.aii_path + '.csv'

        targets = get_targets()
        mimic_pre_processor = MimicPreProcessor(output_filepath)
        pickle_path = get_pickle_folder(mimic_version)

        for target in targets:
            print(f'Creating Datasets for {target}')
            mimic_pre_processor.pre_process_and_save_files(target, window_size, pickle_path)
            print(f'Created Datasets for {target}\n')

    if create_models:
        train_models(get_targets(), get_percentages(), mimic_version)


if __name__ == "__main__":
    parse = True
    pre_process = False
    train = False
    mimic_v = 4
    window_s = 24
    main(parse, pre_process, train, mimic_v, window_s)
