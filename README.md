# Time Series Load Forecasting of Household Energy Consumption

## Project Overview

This project aims to predict hourly household energy consumption using a machine learning model. By accurately forecasting energy usage, we can enable better energy management, optimize resource allocation, and support demand-side planning.

The core of this project is an XGBoost Regressor trained on a time-series dataset. The model leverages extensive feature engineering to capture seasonality, trends, and other temporal patterns, resulting in a low Mean Absolute Error (MAE) and demonstrating high predictive accuracy.

## Tech Stack

Programming Language: Python

Data Manipulation: Pandas, NumPy

Machine Learning: Scikit-learn, XGBoost

Data Visualization: Matplotlib, Seaborn

Environment: Jupyter Notebook

## Dataset

This project uses the "Individual household electric power consumption" dataset from the UCI Machine Learning Repository.

Source: UCI Machine Learning Repository

Description: The dataset contains 2,075,259 measurements of electric power consumption in one household with a one-minute sampling rate over a period of almost 4 years.

Key Columns: Global_active_power, Global_reactive_power, Voltage, Global_intensity, etc.

## Project Workflow

The project is structured in a Jupyter Notebook and follows these key steps:

## 1. Data Preprocessing

Loaded the dataset, parsing the Date and Time columns into a single datetime index.

Handled missing values (common in this dataset) using appropriate imputation techniques (e.g., forward-fill or interpolation).

Resampled the one-minute data into hourly averages to make the forecasting task more manageable and relevant.

## 2. Exploratory Data Analysis (EDA)

Visualized the time series data to identify key patterns, including:

Overall trends (long-term changes in consumption).

Seasonality (annual, weekly, and daily cycles).

Outliers and anomalies in the data.

## 3. Feature Engineering

This was a critical step for model performance. The following features were created from the datetime index:

Calendar Features: hour_of_day, day_of_week, month, quarter, year.

Lag Features: Consumption data from previous time steps (e.g., 24 hours ago, 1 week ago) to capture auto-correlation.

Rolling Window Features: Rolling means (e.g., 7-day rolling average) to smooth out short-term fluctuations and capture trends.

## 4. Model Training & Evaluation

Model: An XGBoost Regressor was selected for its high performance in tabular and time-series-like datasets.

Train-Test Split: The data was split into training and testing sets chronologically to ensure the model was only trained on past data and tested on future data.

Evaluation: The model's performance was evaluated using standard regression metrics, primarily Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).

## Results

The final XGBoost model performed exceptionally well, achieving a low Mean Absolute Error (MAE). This indicates a high level of accuracy in predicting the hourly energy consumption.

A plot of the predicted values versus the actual values on the test set shows the model's ability to capture the complex patterns of energy usage.

(You can insert your Matplotlib graph here as an image)
![Model Performance](plot.png)

## How to Run This Project

To replicate this project on your local machine, follow these steps:

## Clone the repository:

git clone [https://github.com/your-username/household-energy-forecasting.git](https://github.com/your-username/household-energy-forecasting.git)
cd household-energy-forecasting


Create and activate a virtual environment:

## For macOS/Linux
python3 -m venv venv
source venv/bin/activate

## For Windows
python -m venv venv
.\venv\Scripts\activate


Install the required dependencies:

pip install -r requirements.txt


## Download the dataset:

Download the data from the UCI link and place the household_power_consumption.txt file in a data/ directory.

Run the Jupyter Notebook:

jupyter notebook forecasting_analysis.ipynb
