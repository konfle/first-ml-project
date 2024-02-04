import pandas as pd
import re

data = pd.read_csv("rent_apartments.csv")

print(data)

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
