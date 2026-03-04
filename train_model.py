# =====================================================
# BACKEND FROZEN VERSION
# Final Model: Gradient Boosting (Tuned)
# Final R2 Score: 0.9237
# Model File: final_aqi_model.pkl
# DO NOT MODIFY TRAINING LOGIC
# =====================================================

import pandas as pd  # pyre-ignore
import numpy as np  # pyre-ignore
np.random.seed(42)

def get_aqi_category(aqi):
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Moderate"
    elif aqi <= 200:
        return "Unhealthy"
    elif aqi <= 300:
        return "Very Unhealthy"
    else:
        return "Hazardous"

# =====================================================
# 1️⃣ LOAD DATA
# =====================================================

df = pd.read_csv("india_city_aqi_2015_2025_10cities_encoded.csv")

df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(['city','date']).reset_index(drop=True)

# =====================================================
# 2️⃣ ADVANCED FEATURE ENGINEERING
# =====================================================

# Multi-lag features
lags = [3, 7, 14, 30]
for lag in lags:
    df[f'aqi_lag_{lag}'] = df.groupby('city')['aqi'].shift(lag)

# Rolling features
df['rolling_mean_14'] = df.groupby('city')['aqi'].transform(lambda x: x.rolling(14).mean())
df['rolling_std_7'] = df.groupby('city')['aqi'].transform(lambda x: x.rolling(7).std())
df['rolling_std_14'] = df.groupby('city')['aqi'].transform(lambda x: x.rolling(14).std())

# Interaction feature
df['pm25_no2_interaction'] = df['pm25'] * df['no2']

# Monthly avg AQI
df['monthly_avg_aqi'] = df.groupby(['city','year','month'])['aqi'].transform('mean')

# City avg pollution
df['city_avg_pollution'] = df.groupby('city')['pollution_index'].transform('mean')

# Drop NaN from lag/rolling
df = df.dropna().reset_index(drop=True)

print("Feature Engineering Completed")
print("Final Shape:", df.shape)

# =====================================================
# 3️⃣ PREPARE DATA FOR MODELING
# =====================================================

# Select only numeric columns automatically
numeric_df = df.select_dtypes(include=['number'])

# Separate target
X = numeric_df.drop(columns=['aqi'])
y = numeric_df['aqi']

print("Feature Matrix Shape:", X.shape)
print("Target Shape:", y.shape)

# =====================================================
# IMPROVED TIME SPLIT
# =====================================================

# Train on 2015–2024
train = df[df['year'] <= 2024]

# Test only on 2025
test = df[df['year'] == 2025]

X_train = train.select_dtypes(include=['number']).drop(columns=['aqi'])
y_train = train['aqi']

X_test = test.select_dtypes(include=['number']).drop(columns=['aqi'])
y_test = test['aqi']

print("Train Shape:", X_train.shape)
print("Test Shape:", X_test.shape)

# =====================================================
# LINEAR REGRESSION (BASELINE)
# =====================================================

from sklearn.linear_model import LinearRegression  # pyre-ignore
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # pyre-ignore

lr = LinearRegression()
lr.fit(X_train, y_train)

lr_pred = lr.predict(X_test)

lr_mae = mean_absolute_error(y_test, lr_pred)
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
lr_r2 = r2_score(y_test, lr_pred)

print("\nLinear Regression Results")
print("MAE:", lr_mae)
print("RMSE:", lr_rmse)
print("R2:", lr_r2)

# =====================================================
# RANDOM FOREST MODEL (IMPROVED)
# =====================================================

from sklearn.ensemble import RandomForestRegressor  # pyre-ignore

rf = RandomForestRegressor(
    n_estimators=400,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

rf_mae = mean_absolute_error(y_test, rf_pred)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
rf_r2 = r2_score(y_test, rf_pred)

print("\nRandom Forest Results")
print("MAE:", rf_mae)
print("RMSE:", rf_rmse)
print("R2:", rf_r2)
# =====================================================
# FINAL TUNED GRADIENT BOOSTING MODEL
# =====================================================

from sklearn.ensemble import GradientBoostingRegressor  # pyre-ignore

best_gbr = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)

best_gbr.fit(X_train, y_train)

gbr_pred = best_gbr.predict(X_test)

gbr_r2 = r2_score(y_test, gbr_pred)

print("\nFinal Tuned Gradient Boosting R2:", gbr_r2)
# =====================================================
# XGBOOST MODEL
# =====================================================

from xgboost import XGBRegressor  # pyre-ignore

xgb = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    objective='reg:squarederror'
)

xgb.fit(X_train, y_train)

xgb_pred = xgb.predict(X_test)

xgb_mae = mean_absolute_error(y_test, xgb_pred)
xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))
xgb_r2 = r2_score(y_test, xgb_pred)

print("\nXGBoost Results")
print("MAE:", xgb_mae)
print("RMSE:", xgb_rmse)
print("R2:", xgb_r2)

print("\n========== MODEL SUMMARY ==========")
print(f"Linear R2: {lr_r2:.4f}")
print(f"Random Forest R2: {rf_r2:.4f}")
print(f"Tuned Gradient Boosting R2: {gbr_r2:.4f}")
print(f"XGBoost R2: {xgb_r2:.4f}")
print("===================================")

model_scores = {
    "Linear": lr_r2,
    "Random Forest": rf_r2,
    "Gradient Boosting (Tuned)": gbr_r2,
    "XGBoost": xgb_r2
}

best_model_name = max(model_scores, key=lambda k: model_scores[k])

print(f"\nBest Model Selected Automatically: {best_model_name}")

model_info = {
    "Best Model": best_model_name,
    "R2 Score": model_scores[best_model_name],
    "Train Shape": X_train.shape,
    "Test Shape": X_test.shape
}

print("\nModel Information:")
print(model_info)

# =====================================================
# TEST AQI CATEGORY FUNCTION
# =====================================================

sample_prediction = gbr_pred[0]
print("\nSample Predicted AQI:", sample_prediction)
print("Sample AQI Category:", get_aqi_category(sample_prediction))

# =====================================================
# FEATURE IMPORTANCE (Gradient Boosting)
# =====================================================

import pandas as pd  # pyre-ignore

feature_importance = pd.Series(
    best_gbr.feature_importances_,
    index=X_train.columns
).sort_values(ascending=False)

print("\nTop 10 Important Features:")
print(feature_importance.head(10))

import matplotlib.pyplot as plt  # pyre-ignore

plt.figure(figsize=(8,6))
plt.scatter(y_test, gbr_pred, alpha=0.5)
plt.xlabel("Actual AQI")
plt.ylabel("Predicted AQI")
plt.title("Actual vs Predicted AQI (Gradient Boosting)")
plt.show()

import json

feature_columns = list(X_train.columns)

with open("feature_columns.json", "w") as f:
    json.dump(feature_columns, f)
    
import joblib  # pyre-ignore

joblib.dump(best_gbr, "final_aqi_model.pkl")
print("Final model saved successfully.")

sample_data = X_test[:5]

predictions = best_gbr.predict(sample_data)

print("Sample Predictions:")
print(predictions)

# =====================================================
# BACKEND STABILITY VERIFICATION
# =====================================================

print("\n========== BACKEND STABILITY CHECK ==========")

# 1️⃣ Load saved model
loaded_model = joblib.load("final_aqi_model.pkl")
print("Model loaded successfully.")

# 2️⃣ Predict again using loaded model
loaded_pred = loaded_model.predict(X_test)

print("Sample Predictions from Loaded Model:")
print(loaded_pred[:5])

# 3️⃣ Recalculate R2
loaded_r2 = r2_score(y_test, loaded_pred)
print("R2 Score from Loaded Model:", loaded_r2)

# 4️⃣ Verify AQI Category
print("\nAQI Category Test:")
print("AQI:", loaded_pred[0],
      "| Category:", get_aqi_category(loaded_pred[0]))

print("============================================")
