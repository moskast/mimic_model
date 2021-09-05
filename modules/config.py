class AppConfig:
    random_seed = 0
    mimic_version = 4

    window_size = 24

    # The targets for the experiments
    targets = ['VANCOMYCIN', 'MI', 'SEPSIS']
    # List of percentages of training data to be used for the experiments
    percentages = [1.0]

    train_single_targets = True

    create_statistics = False
    balance_data = False
    oversample = False
