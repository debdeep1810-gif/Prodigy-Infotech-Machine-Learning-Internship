import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ============================================
# DATASET PATH
# ============================================

csv_path = r"D:\DEBDEEP\Data\prices.csv"

# ============================================
# CHECK FILE EXISTS
# ============================================

if not os.path.exists(csv_path):
    print("CSV file not found!")
    print("Path:", csv_path)
    exit()

print("CSV file found successfully!")

# ============================================
# LOAD DATASET
# ============================================

df = pd.read_csv(csv_path)

print("\nFirst 5 rows:")
print(df.head())

print("\nColumns:")
print(df.columns.tolist())

# ============================================
# REMOVE NON-NUMERIC COLUMNS
# ============================================

numeric_df = df.select_dtypes(include=['number'])

print("\nNumeric Columns:")
print(numeric_df.columns.tolist())

# ============================================
# TARGET COLUMN
# Predict CLOSE price
# ============================================

X = numeric_df.drop('close', axis=1)

y = numeric_df['close']

# ============================================
# TRAIN TEST SPLIT
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ============================================
# MODEL
# ============================================

model = LinearRegression()

print("\nTraining model...")

model.fit(X_train, y_train)

# ============================================
# PREDICTIONS
# ============================================

y_pred = model.predict(X_test)

# ============================================
# EVALUATION
# ============================================

print("\n===== MODEL PERFORMANCE =====")

print("MAE :", mean_absolute_error(y_test, y_pred))

rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("RMSE:", rmse)

print("R2 Score:", r2_score(y_test, y_pred))

# ============================================
# COEFFICIENTS
# ============================================

print("\nIntercept:")
print(model.intercept_)

print("\nCoefficients:")

for feature, coef in zip(X.columns, model.coef_):
    print(feature, ":", coef)
