# Rossmann Store Sales Prediction

This project implements a machine learning pipeline to forecast daily sales for Rossmann stores using historical data. The solution involves extensive data cleaning, feature engineering, and a gradient-boosted decision tree model (XGBoost).

## Project Overview
Rossmann operates over 3,000 drugstores in 7 European countries. Store sales are influenced by many factors, including promotions, competition, school and state holidays, seasonality, and locality.

## Key Features of the Pipeline
- **Data Integration**: Merged store metadata with daily sales transactions.
- **Missing Value Imputation**: Handled missing values for competition distance and promotion intervals.
- **Feature Engineering**:
  - Extracted temporal features: Year, Month, Day, Week of Year, and Day of Week.
  - Created `CompetitionOpenMonths` to track how long competition has been active.
  - Applied One-Hot Encoding to categorical variables like `StoreType`, `Assortment`, and `StateHoliday`.
- **Modeling**: Utilized `XGBRegressor` with log-transformed targets to minimize the impact of variance in high-sales days.

## Model Performance
- **R² Score**: ~0.9615
- **RMSE**: ~595.06

## ToolBox
- Python 3.x
- Pandas
- NumPy
- XGBoost
- Scikit-learn
- Matplotlib/Seaborn

