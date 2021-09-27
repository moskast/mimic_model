class AppConfig:
    """
    Class containing all application wide variables.
    random_seed: int
        random seed on which to train on
    mimic_version: int
        which mimic version to use
    window_size:
        number of hours in a time step
    targets: list[str]
        targets on which to train on
    train_single_targets: bool
        whether to train on single targets if in multitask setting
    create_statistics: bool
        whether or not to create min, max and std columns
    balance_data: bool
        whether to balance the data during pre processing
    oversample: bool
        whether to oversample the data
    """
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