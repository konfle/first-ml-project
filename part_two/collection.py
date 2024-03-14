import pandas as pd
import os


def load_data(path="../part_one/rent_apartments.csv"):
    """
    Load data from a CSV file.

    Parameters:
    - path (str, optional): The path to the CSV file. Default is "../part_one/rent_apartments.csv".
      If a relative path is provided, it is resolved relative to the script's location.

    Returns:
    - pd.DataFrame: A pandas DataFrame containing the loaded data.
    """
    # Get the absolute path to the script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Change the working directory to the script's directory
    os.chdir(script_dir)

    # Load data from the specified CSV file
    return pd.read_csv(path)
