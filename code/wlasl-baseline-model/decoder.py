import pandas as pd


def get_gloss(num: int) -> str:
    """
    Return gloss given numeric value

    Parameters
    ----------
    num : int
        DESCRIPTION.

    Returns
    -------
    str

    """
    df = pd.read_csv('wlasl_class_list.txt', sep='\t', header=None, names=["num", "word"])
    return df[df['num'] == num]['word'].values[0]
