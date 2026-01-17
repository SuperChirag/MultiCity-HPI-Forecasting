import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score

# 1. LOAD DATA
# Use the file path for your CSV converted from Sheet2
df = pd.read_csv(r"E:\hpi\hi.csv")

# 2. DATA CLEANING & ENCODING 
# Remove rows with missing values (essential for Lag variables)
df_clean = df.dropna().reset_index(drop=True)

# One-hot encode the 10 cities (creates binary flags for each city)
df_encoded = pd.get_dummies(df_clean, columns=['City'], drop_first=True)

# 3. DEFINE FEATURES (X) AND TARGET (y)
X = df_encoded.drop(['HPI@Assessment Prices', 'Quarter'], axis=1)
y = df_encoded['HPI@Assessment Prices']

# 4. TRAIN/TEST SPLIT (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. MODEL 1: LINEAR REGRESSION (The Baseline)
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)

# 6. MODEL 2: GRADIENT BOOSTING (The Advanced Model)
gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42)
gb_model.fit(X_train, y_train)
gb_pred = gb_model.predict(X_test)

# 7. METRIC EVALUATION
lr_r2 = r2_score(y_test, lr_pred)
gb_r2 = r2_score(y_test, gb_pred)

print("--- Accuracy Comparison (R-Squared) ---")
print(f"Linear Regression: {lr_r2:.4f}")
print(f"Gradient Boosting: {gb_r2:.4f}")

# 8. VISUALIZATION (For your project report)
# Plotting the comparison of Accuracy
models = ['Linear Regression', 'Gradient Boosting']
scores = [lr_r2, gb_r2]

plt.figure(figsize=(8, 5))
plt.bar(models, scores, color=['skyblue', 'salmon'])
plt.ylabel('R-Squared Score')
plt.title('Housing Price Index: Model Comparison')
plt.ylim(0, 1.1)
plt.show()