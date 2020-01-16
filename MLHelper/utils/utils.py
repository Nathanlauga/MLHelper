import os

class OpenFile:
    """
    Class that open a file and close it at the end
    Attributes
    ----------
    fname : str
        file name
    mode : str
        mode for open() method
    """
    def __init__(self, fname: str, mode='r'):
        self.fname = fname
        self.mode = mode

    def __enter__(self):
        self.file = open(self.fname, self.mode)
        return self.file

    def __exit__(self, type, value, traceback):
        self.file.close()


def str_to_file(string: str, fname: str):
    """
    Create a file based on a string.
    Parameters
    ----------
    string: str
        string to write into the file
    fname: str
        file name
    """
    fpath = os.getcwd() + '/' + fname

    with OpenFile(fpath, 'w') as file:
        file.write(string)
    file.close()
    print('File created at ', fpath)


def remove_var_with_one_value(df):
    """
    Remove dataset's columns that only contains one unique value.

    Parameters
    ----------
    df: pd.DataFrame
        Dataframe to inspect
    
    Returns
    -------
    pd.DataFrame:
        Dataframe without columns with only one unique value
    """
    if len(df) <= 1:
        return df
    
    for var in df:
        if df[var].nunique() <= 1:
            del df[var]

    return df