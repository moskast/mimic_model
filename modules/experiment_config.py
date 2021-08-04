def get_targets():
    """
    @return: The targets for the experiments
    """
    return ['VANCOMYCIN', 'MI', 'SEPSIS']


def get_percentages():
    """
    @return: Percentages of training data to be used for the experiments
    """
    return [1.0]


def get_seeds():
    return list(range(5))


def get_window_size():
    return 24
    return 12


def get_mimic_version():
    return 4


def get_random_seed():
    return 0


def get_train_comparison():
    return False


def get_train_single_targets():
    return True
