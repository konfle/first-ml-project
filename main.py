import re
import pandas as pd
import pickle as pk

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

# 1. DATA COLLECTION
data = pd.read_csv("rent_apartments.csv")

print(data)

# 2. DATA PREPARATION
print(data.dtypes)

# encode object data types
data_encoded = pd.get_dummies(data, columns=["balcony", "parking", "furnished", "garage", "storage"], drop_first=True)
print(data_encoded)

# garden column
print(data_encoded.garden.unique())

# extract single are of 5th element as integer data type
print(f"\nThe 5th garden area is: {data_encoded.garden[4]}")
area = int(re.findall(pattern='\d+', string=data_encoded.garden[4])[0])
print(f"Just area: {area}")

# Convert strings data type to integer
for i in range(len(data_encoded)):
    if data_encoded.loc[i, "garden"] == "Not present":
        data_encoded.loc[i, "garden"] = 0
    else:
        data_encoded.loc[i, "garden"] = int(re.findall(r'\d+', data_encoded.loc[i, "garden"])[0])

print(f"Garden areas: {data_encoded.garden.unique()}")

# 3. MODEL BUILDING
# 3.1. Defining X (independent) and y (dependent) variables
X = data_encoded[['area',
                  'constraction_year',
                  'bedrooms',
                  'garden',
                  'balcony_yes',
                  'parking_yes',
                  'furnished_yes',
                  'garage_yes',
                  'storage_yes']]
y = data_encoded.rent

print(f"\nIndependent variables: {X}")
print(f"\nDependent variables: {y}")

# 3.2. Split the data set with test size 20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 3.3. Model Building
rf = RandomForestRegressor()
rf.fit(X_train, y_train)
print(f"\nCurrent score: {rf.score(X_test, y_test)}")

# 3.4. Prediction
# area (85 m2),
# constraction_year (2015),
# bedrooms (2),
# garden (20 m2),
# balcony_yes (1 - yes),
# parking_yes (1 - yes),
# furnished_yes(0 - no),
# garage_yes (0 - no),
# storage_yes (1 - yes)
apartment_config = [85, 2015, 2, 20, 1, 1, 0, 0, 1]
price = rf.predict([apartment_config])
print(f"\nPredicted price: {round(price[0], 2)}")

# 3.5. Tuning Hyperparameters (rv_v2)
grid_space = dict(n_estimators=[100, 200, 300], max_depth=[3, 6, 9, 12])
grid = GridSearchCV(rf, param_grid=grid_space, cv=5, scoring="r2")
model_grid = grid.fit(X_train, y_train)
print(f"Best hyperparameters are {model_grid.best_params_}, score={model_grid.best_score_}")
best_hp_price = model_grid.best_estimator_.predict([apartment_config])
print(f"Predicted price with best hyperparameters: {round(best_hp_price[0], 2)}")

# 3.6. Models Management
pk.dump(rf, open('models/rf_v1', 'wb'))
rf_v1 = pk.load(open('models/rf_v1', 'rb'))
price_from_saved_model = rf_v1.predict([apartment_config])
print(f"Predicted price from saved model (rf_v1): {round(price_from_saved_model[0], 2)}")

pk.dump(model_grid, open('models/rf_v2', 'wb'))
rf_v2 = pk.load(open('models/rf_v2', 'rb'))
price_from_saved_model = rf_v2.predict([apartment_config])
print(f"Predicted price from saved model (rf_v2): {round(price_from_saved_model[0], 2)}")
