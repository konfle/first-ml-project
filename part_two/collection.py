import pandas as pd


def load_data(path="../part_one/rent_apartments.csv"):
    """
    Load data from a CSV file.

    Parameters:
    - path (str, optional): The path to the CSV file. Default is "part_one/rent_apartments.csv".

    Returns:
    - pd.DataFrame: A pandas DataFrame containing the loaded data.
    """
    return pd.read_csv(path)

