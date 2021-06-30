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
