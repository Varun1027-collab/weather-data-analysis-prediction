import streamlit as st
import pandas as pd
import joblib

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Weather Data Analysis & Prediction",
    page_icon="🌦️",
    layout="wide"
)

# =========================================================
# LOAD DATA AND MODEL
# =========================================================

df = pd.read_csv("weather_data.csv")

df["date"] = pd.to_datetime(df["date"])

model = joblib.load("weather_model.pkl")

# =========================================================
# TITLE
# =========================================================

st.title("🌦️ Weather Data Analysis & Prediction")

st.write(
    "An AI-based application for analyzing historical weather "
    "data and predicting maximum temperature using Machine Learning."
)

st.divider()

# =========================================================
# DATASET OVERVIEW
# =========================================================

st.subheader("📊 Dataset Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Records",
        f"{len(df):,}"
    )

with col2:
    st.metric(
        "Cities",
        df["city"].nunique()
    )

with col3:
    st.metric(
        "Start Year",
        df["date"].dt.year.min()
    )

with col4:
    st.metric(
        "End Year",
        df["date"].dt.year.max()
    )

st.divider()

# =========================================================
# CITY SELECTION
# =========================================================

st.subheader("🏙️ Select City")

city = st.selectbox(
    "Choose a city",
    sorted(df["city"].unique()),
    key="analysis_city"
)

city_data = df[df["city"] == city].copy()

# =========================================================
# WEATHER ANALYSIS
# =========================================================

st.subheader(f"📈 Weather Analysis — {city}")

col1, col2, col3 = st.columns(3)

with col1:
    avg_max_temp = city_data["temperature_2m_max"].mean()

    st.metric(
        "Average Maximum Temperature",
        f"{avg_max_temp:.1f} °C"
    )

with col2:
    avg_min_temp = city_data["temperature_2m_min"].mean()

    st.metric(
        "Average Minimum Temperature",
        f"{avg_min_temp:.1f} °C"
    )

with col3:
    avg_rain = city_data["rain_sum"].mean()

    st.metric(
        "Average Rainfall",
        f"{avg_rain:.2f} mm"
    )

# =========================================================
# TEMPERATURE TREND
# =========================================================

st.subheader("🌡️ Temperature Trend")

chart_data = city_data[
    [
        "date",
        "temperature_2m_max",
        "temperature_2m_min"
    ]
].copy()

chart_data = chart_data.sort_values("date")

st.line_chart(
    chart_data,
    x="date",
    y=[
        "temperature_2m_max",
        "temperature_2m_min"
    ],
    x_label="Date",
    y_label="Temperature (°C)"
)

st.divider()

# =========================================================
# SMART TEMPERATURE PREDICTION
# =========================================================

st.subheader("🤖 Smart Maximum Temperature Prediction")

st.write(
    "Choose a city and date. You can either use historical weather "
    "conditions from the dataset or enter weather conditions manually."
)

# =========================================================
# PREDICTION CITY AND DATE
# =========================================================

col1, col2 = st.columns(2)

with col1:
    prediction_city = st.selectbox(
        "🏙️ Prediction City",
        sorted(df["city"].unique()),
        key="prediction_city"
    )

with col2:
    prediction_date = st.date_input(
        "📅 Prediction Date",
        value=pd.Timestamp("2024-06-15"),
        min_value=pd.Timestamp("2000-01-01"),
        max_value=pd.Timestamp("2030-12-31")
    )

prediction_timestamp = pd.Timestamp(prediction_date)

# =========================================================
# PREDICTION MODE
# =========================================================

prediction_mode = st.radio(
    "⚙️ Select Prediction Mode",
    [
        "Use Historical Weather Data",
        "Enter Weather Conditions Manually"
    ],
    horizontal=True
)

# =========================================================
# HISTORICAL WEATHER MODE
# =========================================================

if prediction_mode == "Use Historical Weather Data":

    historical_record = df[
        (df["city"] == prediction_city) &
        (df["date"] == prediction_timestamp)
    ]

    if len(historical_record) > 0:

        record = historical_record.iloc[0]

        st.write("### 📋 Weather Data Found")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Minimum Temperature",
                f"{record['temperature_2m_min']:.1f} °C"
            )

        with col2:
            st.metric(
                "Apparent Maximum",
                f"{record['apparent_temperature_max']:.1f} °C"
            )

        with col3:
            st.metric(
                "Rainfall",
                f"{record['rain_sum']:.1f} mm"
            )

        with col4:
            st.metric(
                "Wind Speed",
                f"{record['wind_speed_10m_max']:.1f} km/h"
            )

        st.info(
            "Historical weather conditions were automatically loaded "
            "from the dataset."
        )

        if st.button(
            "🔮 Predict Using Historical Data",
            use_container_width=True
        ):

            input_data = pd.DataFrame({
                "city": [prediction_city],
                "temperature_2m_min": [
                    record["temperature_2m_min"]
                ],
                "apparent_temperature_max": [
                    record["apparent_temperature_max"]
                ],
                "apparent_temperature_min": [
                    record["apparent_temperature_min"]
                ],
                "precipitation_sum": [
                    record["precipitation_sum"]
                ],
                "rain_sum": [
                    record["rain_sum"]
                ],
                "wind_speed_10m_max": [
                    record["wind_speed_10m_max"]
                ],
                "wind_gusts_10m_max": [
                    record["wind_gusts_10m_max"]
                ],
                "wind_direction_10m_dominant": [
                    record["wind_direction_10m_dominant"]
                ],
                "year": [
                    prediction_timestamp.year
                ],
                "month": [
                    prediction_timestamp.month
                ],
                "day": [
                    prediction_timestamp.day
                ]
            })

            prediction = model.predict(
                input_data
            )[0]

            actual_temperature = record[
                "temperature_2m_max"
            ]

            st.success(
                f"🌡️ Predicted Maximum Temperature: "
                f"**{prediction:.2f} °C**"
            )

            st.write(
                f"📌 Actual Maximum Temperature in the dataset: "
                f"**{actual_temperature:.2f} °C**"
            )

            difference = abs(
                prediction - actual_temperature
            )

            st.write(
                f"📊 Difference: **{difference:.2f} °C**"
            )

    else:

        st.warning(
            "No historical weather record was found for this "
            "city and date. Try another date."
        )

# =========================================================
# MANUAL WEATHER MODE
# =========================================================

else:

    st.write("### 🌦️ Enter Weather Conditions")

    # Temperature inputs
    col1, col2, col3 = st.columns(3)

    with col1:
        min_temp = st.number_input(
            "Minimum Temperature (°C)",
            value=20.0,
            step=0.1
        )

    with col2:
        apparent_max = st.number_input(
            "Apparent Maximum Temperature (°C)",
            value=25.0,
            step=0.1
        )

    with col3:
        apparent_min = st.number_input(
            "Apparent Minimum Temperature (°C)",
            value=18.0,
            step=0.1
        )

    # Rain and wind inputs
    col1, col2, col3 = st.columns(3)

    with col1:
        precipitation = st.number_input(
            "Precipitation (mm)",
            min_value=0.0,
            value=0.0,
            step=0.1
        )

    with col2:
        rain = st.number_input(
            "Rain (mm)",
            min_value=0.0,
            value=0.0,
            step=0.1
        )

    with col3:
        wind_speed = st.number_input(
            "Maximum Wind Speed (km/h)",
            min_value=0.0,
            value=10.0,
            step=0.1
        )

    # Wind inputs
    col1, col2 = st.columns(2)

    with col1:
        wind_gust = st.number_input(
            "Maximum Wind Gust (km/h)",
            min_value=0.0,
            value=15.0,
            step=0.1
        )

    with col2:
        wind_direction = st.number_input(
            "Dominant Wind Direction (°)",
            min_value=0.0,
            max_value=360.0,
            value=180.0,
            step=1.0
        )

    if st.button(
        "🔮 Predict Maximum Temperature",
        use_container_width=True
    ):

        input_data = pd.DataFrame({
            "city": [prediction_city],
            "temperature_2m_min": [min_temp],
            "apparent_temperature_max": [apparent_max],
            "apparent_temperature_min": [apparent_min],
            "precipitation_sum": [precipitation],
            "rain_sum": [rain],
            "wind_speed_10m_max": [wind_speed],
            "wind_gusts_10m_max": [wind_gust],
            "wind_direction_10m_dominant": [wind_direction],
            "year": [prediction_timestamp.year],
            "month": [prediction_timestamp.month],
            "day": [prediction_timestamp.day]
        })

        prediction = model.predict(
            input_data
        )[0]

        st.success(
            f"🌡️ Predicted Maximum Temperature in "
            f"**{prediction_city}** on "
            f"**{prediction_timestamp.strftime('%d %B %Y')}**: "
            f"**{prediction:.2f} °C**"
        )

st.divider()

# =========================================================
# HISTORICAL WEATHER DATA
# =========================================================

st.subheader("📋 Historical Weather Data")

st.write(
    f"Recent weather records for **{city}**"
)

recent_data = city_data[
    [
        "date",
        "temperature_2m_max",
        "temperature_2m_min",
        "rain_sum",
        "wind_speed_10m_max"
    ]
].sort_values(
    "date",
    ascending=False
).head(10)

recent_data = recent_data.rename(
    columns={
        "date": "Date",
        "temperature_2m_max": "Max Temperature (°C)",
        "temperature_2m_min": "Min Temperature (°C)",
        "rain_sum": "Rainfall (mm)",
        "wind_speed_10m_max": "Wind Speed (km/h)"
    }
)

st.dataframe(
    recent_data,
    use_container_width=True,
    hide_index=True
)

st.divider()

# =========================================================
# HOW IT WORKS
# =========================================================

st.subheader("🧠 How Does It Work?")

st.write("""
**1. Weather Dataset**  
Historical weather data from Indian cities is loaded from the CSV file.

**2. Data Processing**  
The date is converted into useful features such as year, month and day.

**3. City Processing**  
The selected city is converted into a machine-learning feature using One-Hot Encoding.

**4. Machine Learning**  
A Random Forest Regression model learns patterns from historical weather data.

**5. Prediction**  
The selected city, date and weather conditions are given to the model to predict maximum temperature.

**6. Visualization**  
Historical temperature data is displayed using an interactive temperature trend chart.
""")

st.divider()

# =========================================================
# MACHINE LEARNING MODEL
# =========================================================

st.subheader("📋 Machine Learning Model")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Algorithm",
        "Random Forest"
    )

with col2:
    st.metric(
        "R² Score",
        "0.97"
    )

with col3:
    st.metric(
        "MAE",
        "0.57 °C"
    )

st.divider()

# =========================================================
# FOOTER
# =========================================================

st.caption(
    "🌦️ Weather Data Analysis & Prediction | "
    "Machine Learning Internship Project"
)