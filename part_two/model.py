import pickle as pk
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from preparation import prepare_data


def build_model():
    """
    Build a random forest regression model, train it on the preprocessed dataset, evaluate its performance,
    and save the model.

    Returns:
    - None
    """
    # Load preprocessed dataset
    data_frame = prepare_data()
    # Identify X and y
    x, y = get_x_and_y(data_frame)
    # Split the dataset
    x_train, x_test, y_train, y_test = split_train_test(x, y)
    # Train model
    random_forest_model = train_model(x_train, y_train)
    # Evaluate the model
    score = evaluate_model(random_forest_model, x_test, y_test)
    print(f"Model score: {score}")
    # Save the model
    save_model(random_forest_model)


def get_x_and_y(data, col_x=None, col_y=None):
    """
    Extract feature columns (X) and target column (y) from the provided DataFrame.

    Parameters:
    - data (pd.DataFrame): The input DataFrame.
    - col_x (list, optional): List of feature column names. Default is None, using predefined columns.
    - col_y (str, optional): The target column name. Default is None, using "rent".

    Returns:
    - tuple: A tuple containing feature columns (X) and the target column (y).
    """
    if col_x is None:
        col_x = ['area', 'constraction_year', 'bedrooms',
                 'garden', 'balcony_yes', 'parking_yes',
                 'furnished_yes', 'garage_yes', 'storage_yes']
    if col_y is None:
        col_y = "rent"
    return data[col_x], data[col_y]


def split_train_test(x, y):
    """
    Split the dataset into training and testing sets.

    Parameters:
    - x (pd.DataFrame): Feature columns (X).
    - y (pd.Series): Target column (y).

    Returns:
    - tuple: A tuple containing training and testing sets for feature columns and target column.
    """
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    return x_train, x_test, y_train, y_test


def train_model(x_train, y_train):
    """
    Train a random forest regression model using GridSearchCV for hyperparameter tuning.

    Parameters:
    - x_train (pd.DataFrame): Training set feature columns (X).
    - y_train (pd.Series): Training set target column (y).

    Returns:
    - RandomForestRegressor: The trained random forest regression model.
    """
    grid_space = dict(n_estimators=[100, 200, 300], max_depth=[3, 6, 9, 12])
    grid = GridSearchCV(RandomForestRegressor(), param_grid=grid_space, cv=5, scoring="r2")
    model_grid = grid.fit(x_train, y_train)
    return model_grid.best_estimator_


def evaluate_model(model, x_test, y_test):
    """
    Evaluate the performance of a trained model on a testing set.

    Parameters:
    - model (RandomForestRegressor): The trained random forest regression model.
    - x_test (pd.DataFrame): Testing set feature columns (X).
    - y_test (pd.Series): Testing set target column (y).

    Returns:
    - float: The R-squared score of the model on the testing set.
    """
    return model.score(x_test, y_test)


def save_model(model):
    """
    Save a trained model using pickle.

    Parameters:
    - model: The trained model to be saved.

    Returns:
    - None
    """
    pk.dump(model, open('../part_two/models/rf_v1', 'wb'))
