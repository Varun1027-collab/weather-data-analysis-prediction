import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np
import joblib

# =========================================================
# LOAD DATASET
# =========================================================

df = pd.read_csv("weather_data.csv")

# Convert date
df["date"] = pd.to_datetime(df["date"])

# =========================================================
# CREATE DATE FEATURES
# =========================================================

df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day

# =========================================================
# FEATURES AND TARGET
# =========================================================

features = [
    "city",
    "temperature_2m_min",
    "apparent_temperature_max",
    "apparent_temperature_min",
    "precipitation_sum",
    "rain_sum",
    "wind_speed_10m_max",
    "wind_gusts_10m_max",
    "wind_direction_10m_dominant",
    "year",
    "month",
    "day"
]

target = "temperature_2m_max"

# Remove missing values
data = df[features + [target]].dropna()

X = data[features]
y = data[target]

# =========================================================
# PREPROCESS CITY
# =========================================================

categorical_features = ["city"]

numeric_features = [
    "temperature_2m_min",
    "apparent_temperature_max",
    "apparent_temperature_min",
    "precipitation_sum",
    "rain_sum",
    "wind_speed_10m_max",
    "wind_gusts_10m_max",
    "wind_direction_10m_dominant",
    "year",
    "month",
    "day"
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "city",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ],
    remainder="passthrough"
)

# =========================================================
# RANDOM FOREST MODEL
# =========================================================

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "regressor",
            RandomForestRegressor(
                n_estimators=100,
                random_state=42
            )
        )
    ]
)

# =========================================================
# TRAIN TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================================================
# TRAIN MODEL
# =========================================================

model.fit(X_train, y_train)

# =========================================================
# PREDICTION
# =========================================================

y_pred = model.predict(X_test)

# =========================================================
# MODEL EVALUATION
# =========================================================

mae = mean_absolute_error(y_test, y_pred)

rmse = np.sqrt(
    mean_squared_error(y_test, y_pred)
)

r2 = r2_score(
    y_test,
    y_pred
)

print("Weather Prediction Model")
print("-------------------------")
print("Total records:", len(data))
print("Cities:", data["city"].nunique())
print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R2 Score:", round(r2, 2))

# =========================================================
# SAVE MODEL
# =========================================================

joblib.dump(
    model,
    "weather_model.pkl"
)
