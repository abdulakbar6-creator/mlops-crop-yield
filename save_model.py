import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

df = pd.read_csv("/home/a-akbar/Documents/crop_yield.csv",
                 nrows=50000)

cat_cols = ['Region', 'Soil_Type', 'Crop', 'Weather_Condition']
le = LabelEncoder()
df_encoded = df.copy()
for col in cat_cols:
    df_encoded[col] = le.fit_transform(df[col])

df_encoded['Fertilizer_Used'] = df_encoded['Fertilizer_Used'].astype(int)
df_encoded['Irrigation_Used'] = df_encoded['Irrigation_Used'].astype(int)

X = df_encoded.drop('Yield_tons_per_hectare', axis=1)
y = df_encoded['Yield_tons_per_hectare']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(
    n_estimators=100, max_depth=10,
    min_samples_leaf=5, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

pickle.dump(model, open("model.pkl", "wb"))
print("Model saved as model.pkl ✅")
