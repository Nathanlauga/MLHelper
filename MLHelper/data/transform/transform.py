import pandas as pd

def convert_date_columns(df):
    """
    Automatic convert date columns as object to datetime.
    It follows the next rules:
    * Is detect as an object column
    * Don't return errors into the pandas.to_datetime function
    
    Parameters
    ----------
    df: pandas.DataFrame
        Dataframe to update
        
    Returns
    -------
    pandas.DataFrame:
        Dataframe with date column as datetime type
    """
    categorical_cols = df.select_dtypes('object').columns

    for col in categorical_cols:
        try:
            df[col] = pd.to_datetime(df[col])
        except:
            continue
            
    del categorical_cols
    return df