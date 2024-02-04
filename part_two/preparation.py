import re
import pandas as pd

from collection import load_data


def prepare_data():
    """
    Load, encode categorical columns, and parse the 'garden' column of the dataset.

    Returns:
    - pd.DataFrame: The prepared DataFrame with encoded categorical columns and parsed 'garden' values.
    """
    # Load the dataset
    data = load_data()
    # Encode columns
    data_encoded = encode_cat_cols(data)
    # Parse the garden column
    df = parse_garden_col(data_encoded)
    return df


def encode_cat_cols(data):
    """
    One-hot encode categorical columns in the DataFrame.

    Parameters:
    - data (pd.DataFrame): The input DataFrame.

    Returns:
    - pd.DataFrame: The DataFrame with one-hot encoded categorical columns.
    """
    return pd.get_dummies(data,
                          columns=["balcony", "parking", "furnished", "garage", "storage"],
                          drop_first=True)


def parse_garden_col(data):
    """
    Parse the 'garden' column, replacing 'Not present' with 0 and extracting numeric values.

    Parameters:
    - data (pd.DataFrame): The input DataFrame.

    Returns:
    - pd.DataFrame: The DataFrame with the 'garden' column parsed.
    """
    for i in range(len(data)):
        if data.loc[i, "garden"] == "Not present":
            data.loc[i, "garden"] = 0
        else:
            data.loc[i, "garden"] = int(re.findall(r'\d+', data.loc[i, "garden"])[0])
    return data
