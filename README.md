# WattWise AI — Fresh Streamlit Deployment

## Files in this package
- `app.py` — complete WattWise AI Streamlit application
- `requirements.txt` — pinned Python dependencies

## Deploy on Streamlit Community Cloud
1. Upload `app.py` and `requirements.txt` to the root of your GitHub repository.
2. In Streamlit Community Cloud, create/redeploy the app from that repository.
3. Set the main file path to `app.py`.

## Files needed for the Model Reconstruction tab
Place these three files in the repository root if you want the pre-trained model reconstruction feature:
- `wattwise_random_trained_rf.joblib`
- `wattwise_block_trained_rf.joblib`
- `wattwise_model_config.json`

These are trained artifacts and cannot safely be recreated from the app source alone. Retrieve the exact files from the Google Colab notebook/session where the models were trained. Do not rename or substitute unrelated `.joblib` files.

The other tabs can be used without those trained artifacts. Upload the relevant experiment CSV when prompted.

## Important project limitation
The synthetic anomalies in this project are experimental examples, not confirmed household faults. The app is a research prototype, not a live electricity meter or fault-diagnosis system.
