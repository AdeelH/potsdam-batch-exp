
def identity(*args, **kwargs):
    return


def replace_zeros(t, val=1):
    t[t == 0] = val
    return t


def fbeta(precision, recall, beta=2):
    beta2 = beta ** 2
    return (1 + beta2) * (precision * recall) / ((beta2 * precision) + recall)