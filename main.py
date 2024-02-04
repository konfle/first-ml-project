import pandas as pd
import re

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

print(f"Independent variables: {X}")
print(f"Dependent variables: {y}")
