# WattWise AI
### Feedback-Guided Detection of Persistent Electricity Anomalies Under Changing Household Routines

## Project Overview
WattWise AI is a machine-learning-based application designed to reconstruct missing household electricity readings and examine whether unusual consumption patterns remain detectable after data reconstruction.

The project compares Linear Interpolation and Random Forest models under different missing-data scenarios, including random missing values and consecutive missing blocks.

## Problem Statement
Missing electricity readings can reduce the reliability of household energy analysis. Reconstructing missing values may also affect the visibility of unusual consumption patterns.

WattWise AI investigates how different reconstruction methods perform under changing missing-data conditions and whether unusual consumption patterns can be retained after reconstruction.

## Key Features
- Upload hourly electricity consumption CSV files.
- Reconstruct missing readings using saved Random Forest models.
- Compare reconstruction approaches.
- Detect unusual electricity consumption patterns.
- Visualise evaluation results and comparisons.
- Evaluate reconstruction performance using MAE and RMSE.

## Technologies Used
- Python
- Streamlit
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Joblib

## Machine Learning Models
1. Random Forest – Random Missing Data Model
2. Random Forest – Block Missing Data Model
3. Linear Interpolation – Baseline Reconstruction Method
4. Isolation Forest – Anomaly Detection

## Dataset
The primary experiment uses the UCI Individual Household Electric Power Consumption dataset.

A separate external household electricity dataset is also used to evaluate reconstruction performance under additional missing-data scenarios.

## Evaluation Metrics
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Anomaly detection and retention analysis

Lower MAE and RMSE values indicate lower reconstruction errors.

## Project Limitations
- The main