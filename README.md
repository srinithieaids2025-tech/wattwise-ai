# WattWise AI — Advanced Streamlit Starter

## Included
- `app.py`: Streamlit app with the historical results from the original experiment and a stronger repeated evaluation.
- `requirements.txt`: Python packages.
- `README.md`: setup instructions.

## Important
This app does not overwrite your existing experiment files. It generates new CSVs only when you download them from the app.

The improved evaluation:
- Uses a chronological 70% training / 30% test split.
- Masks values only in the final test period.
- Calculates MAE and RMSE on hidden readings only.
- Repeats masks for each scenario across multiple random seeds.
- Reports synthetic anomaly retention only when originally detectable hidden anomalies exist.

Synthetic anomaly labels must be present for meaningful anomaly-retention evaluation. These simulated events are not real electrical faults.

## Run locally on a Mac

1. Download and unzip the project.
2. Install Python 3.10 or newer if needed.
3. Open Terminal and move into the unzipped folder:
   ```bash
   cd WattWise_AI_Advanced
   ```
4. Create a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
5. Install packages:
   ```bash
   python -m pip install -r requirements.txt
   ```
6. Start Streamlit:
   ```bash
   streamlit run app.py
   ```
7. Streamlit will show a local address, usually `http://localhost:8501`. Open it in your browser.

## Data to upload

Upload your existing `WattWise_experiment_data.csv` from Google Colab. It should contain:
- a timestamp/date-time column (or a saved timestamp index),
- a power column, preferably `power_with_anomalies_kw`,
- `original_power_kw` as the baseline series,
- `synthetic_anomaly` as a boolean or 0/1 label.

If you only have the results CSV and not the experiment data CSV, the Historical results tab still displays the actual original experiment results, but the stronger evaluation cannot run without the experiment data.

## Interpretation cautions

- The original result table contains full-series MAE/RMSE. The new repeated experiment calculates errors on hidden readings only, so its numbers are not directly comparable to the original table.
- The new evaluation is a stronger controlled experiment, not proof of real-world performance.
- One 90-day segment and synthetic anomalies are limited evidence. Test other periods and datasets before making general claims.
