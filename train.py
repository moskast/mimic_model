from modules.classes.mimic_parser import MimicParser
from modules.classes.mimic_pre_processor import MimicPreProcessor
from modules.train_model import train_model


def get_targets():
    return ['MI', 'SEPSIS', 'VANCOMYCIN']


def get_percentages():
    # return [1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.05]
    return [1.0]


def pre_process_mimic(filepath, targets, mimic_version):
    mimic_pre_processor = MimicPreProcessor(filepath)
    output_folder = f'./output/pickled_data_sets/mimic_{mimic_version}'
    for target in targets:
        print(f'Creating Datasets for {target}')
        mimic_pre_processor.pre_process_and_save_files(target, output_folder)
        print(f'Created Datasets for {target}\n')


def train_models(targets, percentages, mimic_version):
    data_path = f'./output/pickled_data_sets/mimic_{mimic_version}'
    for target in targets:
        print(f'\nTraining {target}')
        for percentage in percentages:
            p = int(percentage * 100)
            model_name = f'mimic_{mimic_version}_{target}_{p}_percent'
            train_model(model_name=model_name, target=target, n_percentage=percentage, data_path=data_path)
            print(f'\rFinished training on {percentage * 100}% of data')


def main(parse_mimic, pre_process_data, create_models, mimic_version=4):
    print('Start Program')
    print(f'Mimic Version {mimic_version}')
    mimic_folder_path = f'./data/mimic_{mimic_version}_database'
    output_folder = 'mapped_elements'
    file_name = 'CHARTEVENTS'
    id_col = 'ITEMID'
    label_col = 'LABEL'

    mp = MimicParser(mimic_folder_path, output_folder, file_name, id_col, label_col, mimic_version)

    if parse_mimic:
        print('Parse Mimic Data')
        mp.perform_full_parsing()

    if pre_process_data:
        print('Preprocess Data')
        if mimic_version == 3:
            output_filepath = mp.an_path + '.csv'
        else:
            output_filepath = mp.aii_path + '.csv'

        targets = get_targets()
        pre_process_mimic(output_filepath, targets, mimic_version)

    if create_models:
        train_models(get_targets(), get_percentages(), mimic_version)


if __name__ == "__main__":
    parse = False
    pre_process = False
    train = True
    mimic_v = 4
    main(parse, pre_process, train, mimic_v)
