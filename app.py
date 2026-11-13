import os
from supabase import create_client
import io
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from pathlib import Path
from supabase import create_client
from sklearn.ensemble import RandomForestRegressor, IsolationForest


# ========================================================
# PAGE CONFIGURATION
# ========================================================

st.set_page_config(
    page_title="WattWise AI",
    page_icon="⚡",
    layout="wide"
)

# ---------------- AUTHENTICATION ----------------

def get_config_value(name):
    value = os.getenv(name)
    if value:
        return value
    return None


SUPABASE_URL = get_config_value("SUPABASE_URL")
SUPABASE_KEY = get_config_value("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    st.error("Supabase configuration is missing. Check Render environment variables.")
    st.stop()

# One Supabase client per Streamlit session
if "wattwise_supabase" not in st.session_state:
    st.session_state["wattwise_supabase"] = create_client(
        SUPABASE_URL, SUPABASE_KEY
    )

supabase = st.session_state["wattwise_supabase"]

if "wattwise_user" not in st.session_state:
    st.session_state["wattwise_user"] = None


# Show login and signup before allowing dashboard access
if st.session_state["wattwise_user"] is None:

    st.title("⚡ WattWise AI")
    st.subheader("Welcome! Please log in to continue.")

    login_tab, signup_tab = st.tabs(["Login", "Create Account"])

    with login_tab:
        with st.form("wattwise_login"):
            email = st.text_input("Email address")
            password = st.text_input("Password", type="password")
            login_clicked = st.form_submit_button("Login")

        if login_clicked:
            try:
                response = supabase.auth.sign_in_with_password({
                    "email": email.strip(),
                    "password": password
                })

                if response.user:
                    st.session_state["wattwise_user"] = {
                        "id": response.user.id,
                        "email": response.user.email
                    }
                    st.rerun()
                else:
                    st.error("Login failed. Please check your credentials.")

            except Exception as e:
                st.error(f"Login failed: {e}")

    with signup_tab:
        with st.form("wattwise_signup"):
            new_email = st.text_input("Email address", key="signup_email")
            new_password = st.text_input(
                "Create password", type="password", key="signup_password"
            )
            confirm_password = st.text_input(
                "Confirm password", type="password"
            )
            signup_clicked = st.form_submit_button("Create Account")

        if signup_clicked:
            if not new_email.strip() or not new_password:
                st.error("Please enter your email and password.")
            elif len(new_password) < 6:
                st.error("Password must contain at least 6 characters.")
            elif new_password != confirm_password:
                st.error("Passwords do not match.")
            else:
                try:
                    response = supabase.auth.sign_up({
                        "email": new_email.strip(),
                        "password": new_password
                    })

                    if response.session and response.user:
                        st.session_state["wattwise_user"] = {
                            "id": response.user.id,
                            "email": response.user.email
                        }
                        st.rerun()
                    else:
                        st.success(
                            "Account created! Check your email to confirm "
                            "your account, then log in."
                        )

                except Exception as e:
                    st.error(f"Sign-up failed: {e}")

    st.stop()


# Logged-in user controls
st.sidebar.success(
    f"Logged in: {st.session_state['wattwise_user']['email']}"
)

if st.sidebar.button("Logout"):
    try:
        supabase.auth.sign_out()
    finally:
        st.session_state["wattwise_user"] = None
        st.session_state.pop("wattwise_supabase", None)
        st.rerun()

# ---------------- END AUTHENTICATION ----------------


BASE_DIR = Path(__file__).resolve().parent


# ========================================================
# SUPABASE CONFIGURATION
# ========================================================

try:
    supabase = create_client(
        st.secrets["SUPABASE_URL"],
        st.secrets["SUPABASE_KEY"]
    )
except Exception as e:
    st.error(f"Could not connect to Supabase: {e}")
    st.stop()


# ========================================================
# SESSION INITIALIZATION
# ========================================================

if "user" not in st.session_state:
    st.session_state.user = None

if "access_token" not in st.session_state:
    st.session_state.access_token = None


# ========================================================
# LOGIN AND SIGNUP
# ========================================================

if st.session_state.user is None:

    st.title("⚡ WattWise AI")
    st.subheader("Welcome! Sign in or create your account.")

    login_tab, signup_tab = st.tabs(["Login", "Sign Up"])

    with login_tab:

        email = st.text_input(
            "Email address",
            key="login_email"
        )

        password = st.text_input(
            "Password",
            type="password",
            key="login_password"
        )

        if st.button("Login", type="primary"):

            if not email.strip() or not password:
                st.warning("Please enter your email and password.")

            else:
                try:
                    result = supabase.auth.sign_in_with_password({
                        "email": email.strip(),
                        "password": password
                    })

                    if result.user is not None and result.session is not None:

                        st.session_state.user = result.user
                        st.session_state.access_token = (
                            result.session.access_token
                        )

                        st.success("Login successful!")
                        st.rerun()

                    else:
                        st.error("Login failed: No valid session returned.")

                except Exception as e:
                    st.error(f"Login failed: {e}")

    with signup_tab:

        new_email = st.text_input(
            "Email address",
            key="signup_email"
        )

        new_password = st.text_input(
            "Create password",
            type="password",
            key="signup_password"
        )

        if st.button("Create Account", type="primary"):

            if not new_email.strip() or not new_password:
                st.warning("Please enter an email and password.")

            else:
                try:
                    result = supabase.auth.sign_up({
                        "email": new_email.strip(),
                        "password": new_password
                    })

                    if result.session is not None:

                        st.session_state.user = result.user
                        st.session_state.access_token = (
                            result.session.access_token
                        )

                        st.success("Account created successfully!")
                        st.rerun()

                    else:
                        st.success(
                            "Account created! Please verify your email, "
                            "then return here and log in."
                        )

                except Exception as e:
                    st.error(f"Signup failed: {e}")

    st.stop()


# ========================================================
# LOGGED-IN USER SESSION
# ========================================================

st.sidebar.success(
    f"Logged in: {st.session_state.user.email}"
)

if st.session_state.access_token:

    supabase.postgrest.auth(
        st.session_state.access_token
    )


# ========================================================
# LOGOUT
# ========================================================

if st.sidebar.button("Logout"):

    try:
        supabase.auth.sign_out()
    except Exception:
        pass

    st.session_state.user = None
    st.session_state.access_token = None

    st.rerun()


# ========================================================
# FEEDBACK AND SUGGESTIONS
# ========================================================

with st.expander("💬 Feedback & Suggestions"):

    st.subheader("We value your feedback!")

    with st.form("feedback_form"):

        category = st.selectbox(
            "What would you like to share?",
            [
                "General Feedback",
                "Bug Report",
                "Feature Request"
            ]
        )

        message = st.text_area(
            "Describe your feedback",
            placeholder="Write your feedback here..."
        )

        submitted = st.form_submit_button("Submit Feedback")

        if submitted:

            if len(message.strip()) < 5:

                st.warning("Please enter at least 5 characters.")

            else:

                try:
                    supabase.table("feedback_submissions").insert({
                        "user_id": str(st.session_state.user.id),
                        "category": category,
                        "message": message.strip()
                    }).execute()

                    st.success(
                        "Thank you! Your feedback was submitted."
                    )

                except Exception as e:
                    st.error(f"Could not submit feedback: {e}")


# ========================================================
# HISTORICAL RESULTS
# ========================================================

HISTORICAL_RESULTS = pd.DataFrame([

    ["Random 10%", "Linear interpolation", 0.3909, 0.6114, 0.0, 0.0],
    ["Random 10%", "Random Forest", 0.4273, 0.6907, 0.0, 0.0],

    ["3-hour blocks", "Linear interpolation", 0.5138, 0.6523, np.nan, np.nan],
    ["3-hour blocks", "Random Forest", 0.6618, 0.9668, np.nan, np.nan],

    ["6-hour blocks", "Linear interpolation", 1.0218, 1.4386, 0.0, 0.0],
    ["6-hour blocks", "Random Forest", 0.9914, 1.4079, 0.0, 0.0],

    ["12-hour blocks", "Linear interpolation", 0.8763, 1.2914, 0.0, 0.0],
    ["12-hour blocks", "Random Forest", 0.7738, 1.1355, 0.0, 0.0]

], columns=[
    "Scenario",
    "Method",
    "MAE_kW",
    "RMSE_kW",
    "Threshold retention (%)",
    "Isolation Forest retention (%)"
])

SCENARIOS = [
    "Random 10%",
    "3-hour blocks",
    "6-hour blocks",
    "12-hour blocks"
]


# ========================================================
# DATASET PARSING
# ========================================================

def parse_dataset(uploaded):

    df = pd.read_csv(uploaded)

    if df.empty:
        raise ValueError("The uploaded CSV is empty.")

    timestamp_col = None

    preferred = [
        "timestamp",
        "datetime",
        "date_time",
        "date",
        "time",
        "unnamed: 0"
    ]

    lower_map = {
        str(c).strip().lower(): c
        for c in df.columns
    }

    for name in preferred:

        if name in lower_map:

            candidate = lower_map[name]

            parsed = pd.to_datetime(
                df[candidate],
                errors="coerce"
            )

            if parsed.notna().mean() > 0.8:
                timestamp_col = candidate
                break

    if timestamp_col is None:

        for c in df.columns:

            parsed = pd.to_datetime(
                df[c],
                errors="coerce"
            )

            if parsed.notna().mean() > 0.8:
                timestamp_col = c
                break

    if timestamp_col is None:
        raise ValueError(
            "Could not identify a timestamp column. "
            "Include a timestamp/date-time column."
        )

    df[timestamp_col] = pd.to_datetime(
        df[timestamp_col],
        errors="coerce"
    )

    candidates = [
        "power_with_anomalies_kw",
        "ground_truth",
        "power_kw",
        "global_active_power",
        "power",
        "original_power_kw"
    ]

    lower_map = {
        str(c).strip().lower(): c
        for c in df.columns
    }

    power_col = None

    for name in candidates:

        if name in lower_map and lower_map[name] != timestamp_col:
            power_col = lower_map[name]
            break

    if power_col is None:

        numeric_cols = [
            c for c in df.columns
            if c != timestamp_col
            and pd.to_numeric(
                df[c],
                errors="coerce"
            ).notna().mean() > 0.8
        ]

        if not numeric_cols:
            raise ValueError(
                "Could not identify a numeric power column."
            )

        power_col = numeric_cols[0]

    anomaly_col = None

    for c in df.columns:

        if str(c).strip().lower() in [
            "synthetic_anomaly",
            "anomaly",
            "anomaly_label",
            "is_anomaly"
        ]:
            anomaly_col = c
            break

    original_col = None

    for name in [
        "original_power_kw",
        "original_power",
        "normal_power_kw",
        "base_power_kw"
    ]:

        if name in lower_map and lower_map[name] != power_col:
            original_col = lower_map[name]
            break

    out = pd.DataFrame({

        "timestamp": df[timestamp_col],

        "power": pd.to_numeric(
            df[power_col],
            errors="coerce"
        )

    })

    if original_col:

        out["baseline_power"] = pd.to_numeric(
            df[original_col],
            errors="coerce"
        )

    else:

        out["baseline_power"] = out["power"]

    if anomaly_col:

        raw = df[anomaly_col]

        if raw.dtype == bool:

            out["synthetic_anomaly"] = raw.fillna(False)

        else:

            out["synthetic_anomaly"] = (
                raw.astype(str)
                .str.strip()
                .str.lower()
                .isin(["true", "1", "yes", "anomaly"])
            )

    else:

        out["synthetic_anomaly"] = False

    out = out.dropna(
        subset=["timestamp", "power", "baseline_power"]
    )

    out = (
        out.sort_values("timestamp")
        .drop_duplicates("timestamp")
        .set_index("timestamp")
    )

    if len(out) < 200:
        raise ValueError(
            "Please upload the hourly experimental dataset "
            "(at least 200 valid rows)."
        )

    return (
        out,
        power_col,
        bool(anomaly_col),
        bool(original_col)
    )


# ========================================================
# FEATURE ENGINEERING
# ========================================================

def make_features(series):

    idx = series.index

    filled = (
        series.astype(float)
        .interpolate(limit_direction="both")
        .ffill()
        .bfill()
    )

    features = pd.DataFrame(index=idx)

    features["hour"] = idx.hour
    features["day_of_week"] = idx.dayofweek
    features["month"] = idx.month

    features["lag_1h"] = filled.shift(1)
    features["lag_24h"] = filled.shift(24)
    features["lag_168h"] = filled.shift(168)

    return features.bfill().ffill()


# ========================================================
# MISSING DATA MASK
# ========================================================

def create_mask(n, test_start, scenario, seed):

    rng = np.random.default_rng(seed)

    mask = np.zeros(n, dtype=bool)

    test_positions = np.arange(test_start, n)

    if len(test_positions) < 20:
        raise ValueError("Not enough test-period records.")

    if scenario == "Random 10%":

        count = max(
            1,
            int(round(len(test_positions) * 0.10))
        )

        chosen = rng.choice(
            test_positions,
            size=min(count, len(test_positions)),
            replace=False
        )

        mask[chosen] = True

    else:

        block_len = {
            "3-hour blocks": 3,
            "6-hour blocks": 6,
            "12-hour blocks": 12
        }[scenario]

        blocks_to_place = 10
        placed = 0
        attempts = 0

        while placed < blocks_to_place and attempts < 5000:

            attempts += 1

            start = int(
                rng.integers(
                    test_start,
                    n - block_len + 1
                )
            )

            positions = np.arange(
                start,
                start + block_len
            )

            if not mask[positions].any():

                mask[positions] = True
                placed += 1

    return mask


# ========================================================
# EVALUATION METRICS
# ========================================================

def calculate_metrics(y_true, y_pred):

    err = (
        np.asarray(y_true, dtype=float)
        - np.asarray(y_pred, dtype=float)
    )

    mae = float(np.mean(np.abs(err)))

    rmse = float(
        np.sqrt(np.mean(err ** 2))
    )

    return mae, rmse


# ========================================================
# RUN ONE EXPERIMENT
# ========================================================

def run_one_experiment(
    data,
    scenario,
    seed,
    test_fraction=0.30
):

    n = len(data)

    split = int(n * (1 - test_fraction))

    if split < 169 or (n - split) < 50:

        raise ValueError(
            "Dataset is too short for a 70/30 time split "
            "with weekly lag features."
        )

    truth = data["power"].astype(float).copy()

    baseline = data["baseline_power"].astype(float).copy()

    labels = data["synthetic_anomaly"].astype(bool).copy()

    mask = create_mask(
        n,
        split,
        scenario,
        seed
    )

    observed = truth.copy()

    observed.iloc[np.where(mask)[0]] = np.nan

    linear = (
        observed.interpolate(limit_direction="both")
        .ffill()
        .bfill()
    )

    features = make_features(observed)

    train_idx = np.arange(split)

    train_valid = (
        observed.iloc[train_idx]
        .notna()
        .to_numpy()
    )

    train_rows = train_idx[train_valid]

    rf = RandomForestRegressor(
        n_estimators=150,
        max_depth=12,
        min_samples_leaf=3,
        random_state=seed,
        n_jobs=-1
    )

    rf.fit(
        features.iloc[train_rows],
        observed.iloc[train_rows].to_numpy()
    )

    rf_series = observed.copy()

    test_mask_positions = np.where(mask)[0]

    if len(test_mask_positions):

        rf_series.iloc[test_mask_positions] = rf.predict(
            features.iloc[test_mask_positions]
        )

    hidden_idx = np.where(mask)[0]

    y_true = truth.iloc[hidden_idx].to_numpy()

    lin_pred = linear.iloc[hidden_idx].to_numpy()

    rf_pred = rf_series.iloc[hidden_idx].to_numpy()

    lin_mae, lin_rmse = calculate_metrics(
        y_true,
        lin_pred
    )

    rf_mae, rf_rmse = calculate_metrics(
        y_true,
        rf_pred
    )

    threshold = float(
        baseline.iloc[:split].quantile(0.99)
    )

    true_anomaly = labels.iloc[hidden_idx].to_numpy()

    original_detectable_threshold = (
        true_anomaly
        & (
            truth.iloc[hidden_idx].to_numpy()
            > threshold
        )
    )

    train_if = pd.DataFrame({

        "power": baseline.iloc[:split].to_numpy(),

        "hour": baseline.index[:split].hour,

        "day_of_week": baseline.index[:split].dayofweek

    })

    if_model = IsolationForest(
        n_estimators=200,
        contamination=0.01,
        random_state=seed
    )

    if_model.fit(train_if)

    def if_predict(values, timestamps):

        f = pd.DataFrame({

            "power": np.asarray(values, dtype=float),

            "hour": timestamps.hour,

            "day_of_week": timestamps.dayofweek

        })

        return if_model.predict(f) == -1

    original_if_hidden = if_predict(
        truth.iloc[hidden_idx].to_numpy(),
        truth.index[hidden_idx]
    )

    originally_detectable_if = (
        true_anomaly & original_if_hidden
    )

    rows = []

    for method, pred, mae, rmse in [

        (
            "Linear interpolation",
            lin_pred,
            lin_mae,
            lin_rmse
        ),

        (
            "Random Forest",
            rf_pred,
            rf_mae,
            rf_rmse
        )

    ]:

        threshold_detected_after = pred > threshold

        threshold_den = int(
            original_detectable_threshold.sum()
        )

        threshold_num = int(
            (
                original_detectable_threshold
                & threshold_detected_after
            ).sum()
        )

        threshold_retention = (

            100.0 * threshold_num / threshold_den
            if threshold_den
            else np.nan

        )

        if_detected_after = if_predict(
            pred,
            truth.index[hidden_idx]
        )

        if_den = int(
            originally_detectable_if.sum()
        )

        if_num = int(
            (
                originally_detectable_if
                & if_detected_after
            ).sum()
        )

        if_retention = (

            100.0 * if_num / if_den
            if if_den
            else np.nan

        )

        rows.append({

            "Scenario": scenario,

            "Seed": seed,

            "Method": method,

            "Hidden readings": len(hidden_idx),

            "Hidden synthetic anomaly hours": int(
                true_anomaly.sum()
            ),

            "MAE_hidden_kW": mae,

            "RMSE_hidden_kW": rmse,

            "Threshold originally detectable hidden anomalies": threshold_den,

            "Threshold anomalies retained": threshold_num,

            "Threshold retention (%)": threshold_retention,

            "IF originally detectable hidden anomalies": if_den,

            "IF anomalies retained": if_num,

            "Isolation Forest retention (%)": if_retention,

            "Threshold_kW": threshold

        })

    return rows


# ========================================================
# DASHBOARD HEADER
# ========================================================

st.title("⚡ WattWise AI")

st.subheader(
    "Anomaly-Preserving Recovery of Household Electricity Data"
)

st.write(
    "An experimental application comparing missing-data "
    "recovery methods and checking whether synthetic "
    "high-consumption anomalies remain detectable."
)

st.info(
    "Research prototype: synthetic anomalies are simulated "
    "examples, not verified electrical faults. This application "
    "is not a safety or fault-diagnosis system."
)


# ========================================================
# ORIGINAL EXPERIMENT INFORMATION
# ========================================================

with st.expander(
    "About the original experiment",
    expanded=False
):

    st.markdown("""
    The initial experiment used 2,160 hourly records from one
    continuous 90-day period. It introduced eight synthetic
    four-hour episodes by adding 2.5 kW.

    The earlier full-series results are shown below as
    historical results. The improved evaluation separately
    evaluates hidden readings in a chronological holdout period
    and repeats masks across several random seeds.
    """)


# ========================================================
# DASHBOARD TABS
# ========================================================

tab1, tab2, tab3, tab4 = st.tabs([

    "Historical results",

    "Run stronger evaluation",

    "Dataset & limitations",

    "Model Reconstruction"

])


# ========================================================
# TAB 1 - HISTORICAL RESULTS
# ========================================================

with tab1:

    st.markdown("### Original experiment results")

    st.caption(
        "These values are the results from the earlier "
        "Colab experiment. MAE/RMSE were calculated over "
        "the full series, not only hidden readings."
    )

    st.dataframe(
        HISTORICAL_RESULTS,
        use_container_width=True,
        hide_index=True
    )

    fig, ax = plt.subplots(figsize=(9, 4))

    x = np.arange(len(SCENARIOS))

    width = 0.35

    for i, method in enumerate([

        "Linear interpolation",

        "Random Forest"

    ]):

        vals = []

        for scenario in SCENARIOS:

            vals.append(

                HISTORICAL_RESULTS[

                    (HISTORICAL_RESULTS["Scenario"] == scenario)
                    & (HISTORICAL_RESULTS["Method"] == method)

                ]["RMSE_kW"].iloc[0]

            )

        ax.bar(
            x + (i - 0.5) * width,
            vals,
            width,
            label=method
        )

    ax.set_xticks(x)

    ax.set_xticklabels(
        SCENARIOS,
        rotation=12,
        ha="right"
    )

    ax.set_ylabel("Full-series RMSE (kW)")

    ax.set_title(
        "Original experiment: reconstruction error"
    )

    ax.legend()

    st.pyplot(fig)

    plt.close(fig)


# ========================================================
# TAB 2 - STRONGER EVALUATION
# ========================================================

with tab2:

    st.markdown(
        "### Run a stronger, time-aware evaluation"
    )

    st.write(
        "Upload WattWise_experiment_data.csv from your "
        "existing Colab project. The app keeps your "
        "original files unchanged. New outputs are "
        "generated separately."
    )

    uploaded = st.file_uploader(
        "Upload WattWise_experiment_data.csv",
        type=["csv"],
        key="experiment_upload"
    )

    repeats = st.slider(
        "Number of repeated masks per scenario",
        3,
        20,
        10
    )

    base_seed = st.number_input(
        "Base random seed",
        min_value=0,
        max_value=100000,
        value=42,
        step=1
    )

    if uploaded is not None:

        try:

            data, power_col, has_labels, has_baseline = (
                parse_dataset(uploaded)
            )

            st.success(
                f"Loaded {len(data):,} rows. "
                f"Power column detected: `{power_col}`"
            )

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Hourly rows",
                f"{len(data):,}"
            )

            c2.metric(
                "Synthetic anomaly hours",
                f"{int(data['synthetic_anomaly'].sum()):,}"
            )

            c3.metric(
                "Time span",
                f"{(data.index.max() - data.index.min()).days + 1} days"
            )

            if not has_labels:

                st.warning(
                    "No anomaly-label column was found. "
                    "Reconstruction evaluation will work, "
                    "but synthetic-anomaly retention will not "
                    "be meaningful until labels are supplied."
                )

            if not has_baseline:

                st.warning(
                    "No separate original/baseline power "
                    "column was found. The app will use the "
                    "selected power series as the baseline; "
                    "this weakens the anomaly evaluation."
                )

            if not data.index.is_monotonic_increasing:

                st.warning(
                    "Timestamps were sorted during loading."
                )

            st.caption("Preview of the parsed data")

            st.dataframe(
                data.head(8),
                use_container_width=True
            )

            if st.button(
                "Run repeated evaluation",
                type="primary"
            ):

                all_rows = []

                progress = st.progress(0)

                total = repeats * len(SCENARIOS)

                done = 0

                for scenario in SCENARIOS:

                    for rep in range(repeats):

                        seed = int(base_seed) + rep

                        try:

                            all_rows.extend(
                                run_one_experiment(
                                    data,
                                    scenario,
                                    seed
                                )
                            )

                        except Exception as e:

                            st.error(
                                f"Failed for {scenario}, "
                                f"seed {seed}: {e}"
                            )

                            st.stop()

                        done += 1

                        progress.progress(
                            done / total
                        )

                raw_results = pd.DataFrame(all_rows)

                st.session_state[
                    "wattwise_repeat_results"
                ] = raw_results

                summary = raw_results.groupby(
                    ["Scenario", "Method"],
                    as_index=False
                ).agg(

                    Mean_MAE_hidden_kW=(
                        "MAE_hidden_kW",
                        "mean"
                    ),

                    SD_MAE_hidden_kW=(
                        "MAE_hidden_kW",
                        "std"
                    ),

                    Mean_RMSE_hidden_kW=(
                        "RMSE_hidden_kW",
                        "mean"
                    ),

                    SD_RMSE_hidden_kW=(
                        "RMSE_hidden_kW",
                        "std"
                    ),

                    Mean_threshold_retention=(
                        "Threshold retention (%)",
                        "mean"
                    ),

                    Mean_IF_retention=(
                        "Isolation Forest retention (%)",
                        "mean"
                    ),

                    Mean_hidden_readings=(
                        "Hidden readings",
                        "mean"
                    ),

                    Mean_hidden_anomalies=(
                        "Hidden synthetic anomaly hours",
                        "mean"
                    )

                )

                st.session_state[
                    "wattwise_repeat_summary"
                ] = summary

            if "wattwise_repeat_summary" in st.session_state:

                summary = st.session_state[
                    "wattwise_repeat_summary"
                ]

                raw_results = st.session_state[
                    "wattwise_repeat_results"
                ]

                st.markdown(
                    "### Repeated evaluation summary"
                )

                st.caption(
                    "Errors are calculated only on deliberately "
                    "hidden readings in the final 30% chronological "
                    "holdout. SD is the standard deviation across "
                    "repeated masks."
                )

                st.dataframe(
                    summary.round(4),
                    use_container_width=True,
                    hide_index=True
                )

                fig, ax = plt.subplots(
                    figsize=(9, 4)
                )

                x = np.arange(len(SCENARIOS))

                width = 0.35

                for i, method in enumerate([

                    "Linear interpolation",

                    "Random Forest"

                ]):

                    vals = []

                    for scenario in SCENARIOS:

                        hit = summary[

                            (summary["Scenario"] == scenario)
                            & (summary["Method"] == method)

                        ]

                        vals.append(

                            float(
                                hit["Mean_RMSE_hidden_kW"].iloc[0]
                            )
                            if len(hit)
                            else np.nan

                        )

                    ax.bar(
                        x + (i - 0.5) * width,
                        vals,
                        width,
                        label=method
                    )

                ax.set_xticks(x)

                ax.set_xticklabels(
                    SCENARIOS,
                    rotation=12,
                    ha="right"
                )

                ax.set_ylabel(
                    "Mean hidden-only RMSE (kW)"
                )

                ax.set_title(
                    "Repeated time-aware evaluation"
                )

                ax.legend()

                st.pyplot(fig)

                plt.close(fig)

                st.markdown(
                    "### Download your new evaluation files"
                )

                st.download_button(

                    "Download repeated results CSV",

                    data=raw_results.to_csv(
                        index=False
                    ).encode("utf-8"),

                    file_name=(
                        "WattWise_repeated_hidden_only_results.csv"
                    ),

                    mime="text/csv"

                )

                st.download_button(

                    "Download summary CSV",

                    data=summary.to_csv(
                        index=False
                    ).encode("utf-8"),

                    file_name=(
                        "WattWise_repeated_evaluation_summary.csv"
                    ),

                    mime="text/csv"

                )

            else:

                st.caption(
                    "Upload the experiment CSV, then click "
                    "Run repeated evaluation."
                )

        except Exception as e:

            st.error(
                f"Could not load the experiment dataset: {e}"
            )

    else:

        st.warning(
            "Upload the existing experiment CSV to run "
            "the stronger evaluation."
        )


# ========================================================
# TAB 3 - DATASET AND LIMITATIONS
# ========================================================

with tab3:

    st.markdown("### Expected dataset columns")

    st.write(
        "The app looks for a timestamp, a power column "
        "such as `power_with_anomalies_kw`, an optional "
        "`original_power_kw` baseline, and an optional "
        "`synthetic_anomaly` label."
    )

    st.markdown("### Limitations to report")

    st.markdown("""
    - The initial experiment covers one continuous 90-day period with 2,160 hourly records.
    - Eight high-consumption episodes were simulated by adding 2.5 kW for four hours each.
    - Synthetic anomalies are not verified real faults or safety events.
    - The original experiment used one set of missingness masks; the improved evaluation repeats masks.
    - The improved evaluation uses a chronological 70/30 split and calculates errors only on hidden readings.
    - The evaluation is still a controlled experiment on one household and does not prove generalization to other homes.
    - The current application does not connect to live smart meters or issue real-time safety alerts.
    """)

    st.markdown("### Responsible interpretation")

    st.write(
        "Call outputs 'potentially unusual consumption' "
        "or 'synthetic anomaly detections'. Do not claim "
        "that an alert confirms a fault, fire risk, "
        "equipment failure or energy theft."
    )


# ========================================================
# TAB 4 - MODEL RECONSTRUCTION
# ========================================================

with tab4:

    st.markdown(
        "### Recover missing electricity readings"
    )

    st.write(
        "Upload a CSV containing hourly timestamps and "
        "electricity readings. Choose one of the saved "
        "Random Forest models to estimate missing values."
    )

    st.info(
        "This is an offline historical-data reconstruction "
        "tool. Estimates are not actual meter readings, "
        "and unusual values are not confirmed electrical faults."
    )

    @st.cache_resource
    def load_wattwise_models():

        random_model = joblib.load(
            BASE_DIR / "wattwise_random_trained_rf.joblib"
        )

        block_model = joblib.load(
            BASE_DIR / "wattwise_block_trained_rf.joblib"
        )

        with open(
            BASE_DIR / "wattwise_model_config.json",
            "r",
            encoding="utf-8"
        ) as f:

            config = json.load(f)

        return (
            random_model,
            block_model,
            config
        )

    try:

        random_model, block_model, model_config = (
            load_wattwise_models()
        )

        st.success(
            "Saved reconstruction models loaded successfully."
        )

    except Exception as e:

        random_model = None
        block_model = None
        model_config = None

        st.error(
            "Could not load the saved models. Check that "
            "these files are in the GitHub repository root: "
            "wattwise_random_trained_rf.joblib, "
            "wattwise_block_trained_rf.joblib and "
            "wattwise_model_config.json. "
            f"Details: {e}"
        )

    uploaded_reconstruction = st.file_uploader(

        "Upload hourly electricity data (CSV)",

        type=["csv"],

        key="reconstruction_upload"

    )

    if uploaded_reconstruction is not None:

        try:

            reconstruction_df = pd.read_csv(
                uploaded_reconstruction
            )

            if reconstruction_df.empty:

                st.error(
                    "The uploaded CSV is empty."
                )

            else:

                st.write("CSV preview")

                st.dataframe(
                    reconstruction_df.head(),
                    use_container_width=True
                )

                timestamp_options = list(
                    reconstruction_df.columns
                )

                power_options = [

                    c for c in reconstruction_df.columns

                    if pd.to_numeric(
                        reconstruction_df[c],
                        errors="coerce"
                    ).notna().any()

                ]

                timestamp_guess = next(

                    (
                        c for c in timestamp_options

                        if str(c).strip().lower() in [
                            "timestamp",
                            "datetime",
                            "date_time",
                            "date",
                            "time"
                        ]

                    ),

                    timestamp_options[0]

                )

                power_guess = next(

                    (
                        c for c in power_options

                        if str(c).strip().lower() in [
                            "power_kw",
                            "original_power_kw",
                            "power",
                            "global_active_power"
                        ]

                    ),

                    power_options[0]
                    if power_options
                    else None

                )

                timestamp_col = st.selectbox(

                    "Timestamp column",

                    timestamp_options,

                    index=timestamp_options.index(
                        timestamp_guess
                    )

                )

                if not power_options:

                    st.error(
                        "No numeric electricity-reading "
                        "column was found."
                    )

                else:

                    power_col = st.selectbox(

                        "Power column",

                        power_options,

                        index=power_options.index(
                            power_guess
                        )

                    )

                    chosen_model = st.selectbox(

                        "Choose reconstruction model",

                        [
                            "Random-trained Random Forest",
                            "Block-trained Random Forest"
                        ]

                    )

                    if st.button(
                        "Reconstruct missing readings",
                        type="primary"
                    ):

                        if (
                            random_model is None
                            or block_model is None
                        ):

                            st.error(
                                "Models are not loaded. "
                                "Check the model files first."
                            )

                        else:

                            work = reconstruction_df[
                                [timestamp_col, power_col]
                            ].copy()

                            work.columns = [
                                "timestamp",
                                "observed_power_kw"
                            ]

                            work["timestamp"] = pd.to_datetime(
                                work["timestamp"],
                                errors="coerce"
                            )

                            work["observed_power_kw"] = pd.to_numeric(
                                work["observed_power_kw"],
                                errors="coerce"
                            )

                            work = (

                                work.dropna(
                                    subset=["timestamp"]
                                )

                                .sort_values("timestamp")

                                .drop_duplicates("timestamp")

                                .reset_index(drop=True)

                            )

                            missing_mask = (
                                work["observed_power_kw"].isna()
                            )

                            if len(work) < 169:

                                st.error(
                                    "Please provide at least "
                                    "169 hourly rows for the "
                                    "weekly lag features used "
                                    "by this model."
                                )

                            elif not missing_mask.any():

                                st.info(
                                    "No missing readings were "
                                    "found in the selected column."
                                )

                            else:

                                series = pd.Series(

                                    work[
                                        "observed_power_kw"
                                    ].to_numpy(dtype=float),

                                    index=pd.DatetimeIndex(
                                        work["timestamp"]
                                    )

                                )

                                feature_values = (

                                    series.interpolate(
                                        limit_direction="both"
                                    )

                                    .ffill()

                                    .bfill()

                                )

                                features = pd.DataFrame(
                                    index=series.index
                                )

                                features["hour"] = (
                                    series.index.hour
                                )

                                features["day_of_week"] = (
                                    series.index.dayofweek
                                )

                                features["month"] = (
                                    series.index.month
                                )

                                features["lag_1"] = (
                                    feature_values.shift(1)
                                )

                                features["lag_24"] = (
                                    feature_values.shift(24)
                                )

                                features["lag_168"] = (
                                    feature_values.shift(168)
                                )

                                features = (
                                    features.bfill().ffill()
                                )

                                selected_model = (

                                    random_model

                                    if chosen_model
                                    == "Random-trained Random Forest"

                                    else block_model

                                )

                                missing_positions = np.flatnonzero(
                                    missing_mask.to_numpy()
                                )

                                predictions = selected_model.predict(
                                    features.iloc[
                                        missing_positions
                                    ].to_numpy()
                                )

                                work["reconstructed_power"] = (
                                    work["observed_power_kw"]
                                )

                                work.loc[
                                    missing_mask,
                                    "reconstructed_power"
                                ] = predictions

                                st.success(
                                    f"Reconstructed "
                                    f"{int(missing_mask.sum())} "
                                    f"missing readings."
                                )

                                st.caption(
                                    "Existing readings are preserved. "
                                    "Reconstructed values are model "
                                    "estimates, not actual measured "
                                    "consumption."
                                )

                                st.dataframe(

                                    work.loc[missing_mask],

                                    use_container_width=True,

                                    hide_index=True

                                )

                                st.download_button(

                                    "Download reconstructed CSV",

                                    data=work.to_csv(
                                        index=False
                                    ).encode("utf-8"),

                                    file_name=(
                                        "WattWise_reconstructed_readings.csv"
                                    ),

                                    mime="text/csv"

                                )

        except Exception as e:

            st.error(
                f"Could not process the uploaded CSV: {e}"
            )


# ========================================================
# FOOTER
# ========================================================

st.divider()

st.caption(
    "WattWise AI • College research prototype • "
    "Keep the original CSV files unchanged."
)
