def threshold_bipolar(value):
    return 1 if value >= 0 else -1


def threshold_unipolar(value):
    return 1 if value >= 0 else 0
