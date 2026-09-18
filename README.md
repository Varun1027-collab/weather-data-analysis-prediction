# 🌦️ Weather Data Analysis & Prediction

An AI-based machine learning application for analyzing historical weather data from Indian cities and predicting maximum temperature.

## 📌 Project Overview

This project uses historical daily weather data from Indian cities to analyze temperature, rainfall and wind conditions.

A **Random Forest Regression** model is used to predict the maximum temperature based on:

* City
* Minimum temperature
* Apparent temperature
* Rainfall
* Precipitation
* Wind speed
* Wind gusts
* Wind direction
* Year
* Month
* Day

The project provides an interactive **Streamlit web application** for weather analysis and prediction.

## ✨ Features

* 📊 Dataset overview
* 🏙️ Analysis for 10 Indian cities
* 🌡️ Maximum and minimum temperature analysis
* 📈 Historical temperature trend visualization
* 🌧️ Rainfall analysis
* 🤖 Maximum temperature prediction
* 📅 Date-based prediction
* 🏙️ City-based prediction
* 📋 Historical weather data table
* 🔄 Historical weather prediction mode
* ✏️ Manual weather prediction mode
* 📊 Actual vs predicted temperature comparison

## 📂 Dataset

The project uses historical daily weather data covering:

* **Records:** 91,320
* **Cities:** 10
* **Time Period:** 2000–2024

The dataset contains temperature, rainfall, precipitation and wind-related weather information.

## 🧠 Machine Learning

### Algorithm

**Random Forest Regression**

Random Forest uses multiple decision trees and combines their predictions to produce the final temperature prediction.

### Model Performance

The model was evaluated using test data.

| Metric   | Result |
| -------- | -----: |
| R² Score |   0.97 |
| MAE      | 0.57°C |
| RMSE     | 0.77°C |

**MAE (Mean Absolute Error)** represents the average difference between predicted and actual maximum temperature.

**R² Score** measures how well the model explains variation in the target temperature.

These results are based on the project's test split and should not be interpreted as a guarantee of real-world forecast accuracy.

## 🔄 Project Workflow

```text
Weather Dataset
       ↓
Data Preprocessing
       ↓
Date Feature Extraction
       ↓
City Encoding
       ↓
Train/Test Split
       ↓
Random Forest Regression
       ↓
Model Evaluation
       ↓
Save Trained Model
       ↓
Streamlit Application
       ↓
Temperature Prediction
```

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Streamlit
* Joblib

## 📁 Project Structure

```text
CWeather_Data_Analysis_Prediction
│
├── app.py
├── weather_prediction.py
├── weather_data.csv
├── weather_model.pkl
├── requirements.txt
└── README.md
```

## ▶️ How to Run

### 1. Clone the repository

```bash
git clone <repository-url>
```

### 2. Open the project folder

```bash
cd CWeather_Data_Analysis_Prediction
```

### 3. Install required libraries

```bash
pip install -r requirements.txt
```

### 4. Train the model

```bash
python weather_prediction.py
```

This creates:

```text
weather_model.pkl
```

### 5. Start the Streamlit application

```bash
streamlit run app.py
```

### 6. Open the application

Open the local Streamlit URL shown in the terminal, usually:

```text
http://localhost:8501
```

## 🔮 Prediction Modes

### 1. Historical Weather Data

Select a city and historical date. The application automatically retrieves the weather conditions from the dataset and predicts the maximum temperature.

The application also displays:

* Predicted temperature
* Actual temperature
* Difference between predicted and actual temperature

### 2. Manual Weather Conditions

Weather conditions can be entered manually to generate a maximum temperature prediction.

## 📌 Example

For a selected historical city and date, the application may display:

```text
Predicted Maximum Temperature: 43.04°C
Actual Maximum Temperature: 43.60°C
Difference: 0.56°C
```

## 🎯 Project Objective

The main objective is to demonstrate how machine learning can be applied to historical weather data for:

* Weather data analysis
* Temperature pattern identification
* City-based analysis
* Machine learning prediction
* Interactive data visualization

## ⚠️ Note

The application is an internship-level machine learning project based on historical weather data. Predictions depend on the quality and range of the available dataset and should not be treated as official weather forecasts.

## 👨‍💻 Project Type

**Machine Learning Internship Project**

**Domain:** Data Science / Machine Learning / Weather Analytics
