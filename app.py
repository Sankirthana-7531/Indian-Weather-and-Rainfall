import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="Rainfall Prediction Dashboard",
    page_icon="🌧️",
    layout="wide"
)

# ---------------------------------------------------------
# TITLE
# ---------------------------------------------------------

st.title("🌧️ Rainfall Prediction Dashboard")
st.markdown(
    """
    This application predicts whether **rainfall will occur tomorrow**
    based on weather and geographical parameters.
    
    **Rainfall threshold:** 2.5 mm
    """
)

# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

@st.cache_data
def load_data(uploaded_file):
    df = pd.read_csv(uploaded_file)

    # Convert date
    df["date_of_record"] = pd.to_datetime(
        df["date_of_record"],
        errors="coerce"
    )

    # Extract date features
    df["year"] = df["date_of_record"].dt.year
    df["month"] = df["date_of_record"].dt.month
    df["day"] = df["date_of_record"].dt.day

    # Convert rainfall to numeric
    df["rainfall"] = pd.to_numeric(
        df["rainfall"],
        errors="coerce"
    )

    # Remove duplicates
    df = df.drop_duplicates()

    # Fill numerical missing values
    numeric_cols = df.select_dtypes(
        include=np.number
    ).columns

    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    # Fill categorical missing values
    categorical_cols = df.select_dtypes(
        include="object"
    ).columns

    for col in categorical_cols:
        if df[col].notna().any():
            df[col] = df[col].fillna(
                df[col].mode()[0]
            )

    # Rain today
    df["Rain_Today"] = (
        df["rainfall"] >= 2.5
    ).astype(int)

    # Sort by station and date
    df = df.sort_values(
        ["station_name", "date_of_record"]
    )

    # Create tomorrow rainfall target
    df["Rainfall_Tomorrow"] = (
        df.groupby("station_name")["rainfall"]
        .shift(-1)
    )

    # Convert to classification target
    df["Rainfall_Tomorrow"] = (
        df["Rainfall_Tomorrow"] >= 2.5
    ).astype(int)

    return df


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

st.sidebar.header("⚙️ Settings")

uploaded_file = st.sidebar.file_uploader(
    "Upload rainfall CSV file",
    type=["csv"]
)

if uploaded_file is None:
    st.info(
        "Please upload your rainfall CSV dataset using "
        "the sidebar."
    )

    st.markdown(
        """
        ### Expected columns

        The dataset should contain:

        - `date_of_record`
        - `station_name`
        - `rainfall`
        - `avg_temp`
        - `min_temp`
        - `max_temp`
        - `wind_speed`
        - `air_pressure`
        - `latitude`
        - `longitude`
        - `elevation`
        """
    )

    st.stop()


# ---------------------------------------------------------
# LOAD DATASET
# ---------------------------------------------------------

df = load_data(uploaded_file)

st.success(
    f"Dataset loaded successfully — {len(df):,} records"
)


# ---------------------------------------------------------
# FEATURES
# ---------------------------------------------------------

features = [
    "avg_temp",
    "min_temp",
    "max_temp",
    "wind_speed",
    "air_pressure",
    "latitude",
    "longitude",
    "elevation",
    "month"
]

missing_features = [
    col for col in features
    if col not in df.columns
]

if missing_features:
    st.error(
        f"Missing required columns: {missing_features}"
    )
    st.stop()


# ---------------------------------------------------------
# DATA OVERVIEW
# ---------------------------------------------------------

st.header("📊 Dataset Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Records",
        f"{len(df):,}"
    )

with col2:
    st.metric(
        "Features",
        len(features)
    )

with col3:
    st.metric(
        "Rainfall ≥ 2.5 mm",
        f"{(df['rainfall'] >= 2.5).sum():,}"
    )

with col4:
    st.metric(
        "Average Rainfall",
        f"{df['rainfall'].mean():.2f} mm"
    )


# ---------------------------------------------------------
# DATA PREVIEW
# ---------------------------------------------------------

with st.expander("🔎 View Dataset"):
    st.dataframe(
        df.head(100),
        use_container_width=True
    )


# ---------------------------------------------------------
# EDA CHARTS
# ---------------------------------------------------------

st.header("📈 Exploratory Data Analysis")


# ---------------------------------------------------------
# RAINFALL DISTRIBUTION
# ---------------------------------------------------------

st.subheader("1. Rainfall Distribution")

fig, ax = plt.subplots(figsize=(10, 5))

ax.hist(
    df["rainfall"].dropna(),
    bins=50,
    edgecolor="black"
)

ax.set_title("Distribution of Rainfall")
ax.set_xlabel("Rainfall (mm)")
ax.set_ylabel("Frequency")

st.pyplot(fig)


# ---------------------------------------------------------
# MONTHLY RAINFALL
# ---------------------------------------------------------

st.subheader("2. Average Monthly Rainfall")

monthly = (
    df.groupby("month")["rainfall"]
    .mean()
    .reset_index()
)

fig, ax = plt.subplots(figsize=(10, 5))

ax.bar(
    monthly["month"],
    monthly["rainfall"]
)

ax.set_title("Average Monthly Rainfall")
ax.set_xlabel("Month")
ax.set_ylabel("Average Rainfall (mm)")
ax.set_xticks(range(1, 13))

st.pyplot(fig)


# ---------------------------------------------------------
# YEARLY RAINFALL
# ---------------------------------------------------------

st.subheader("3. Average Yearly Rainfall")

yearly = (
    df.groupby("year")["rainfall"]
    .mean()
    .reset_index()
)

fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(
    yearly["year"],
    yearly["rainfall"],
    marker="o"
)

ax.set_title("Average Yearly Rainfall")
ax.set_xlabel("Year")
ax.set_ylabel("Average Rainfall (mm)")

ax.grid(True)

st.pyplot(fig)


# ---------------------------------------------------------
# CORRELATION HEATMAP
# ---------------------------------------------------------

st.subheader("4. Correlation Heatmap")

numeric_df = df.select_dtypes(
    include=np.number
)

fig, ax = plt.subplots(
    figsize=(12, 8)
)

sns.heatmap(
    numeric_df.corr(),
    annot=True,
    fmt=".2f",
    ax=ax
)

ax.set_title(
    "Correlation Heatmap"
)

st.pyplot(fig)


# ---------------------------------------------------------
# TARGET DISTRIBUTION
# ---------------------------------------------------------

st.subheader(
    "5. Rainfall Tomorrow Distribution"
)

target_counts = (
    df["Rainfall_Tomorrow"]
    .value_counts()
    .sort_index()
)

fig, ax = plt.subplots(
    figsize=(8, 5)
)

labels = [
    "No Rain",
    "Rain"
]

values = [
    target_counts.get(0, 0),
    target_counts.get(1, 0)
]

ax.bar(
    labels,
    values
)

ax.set_title(
    "Rainfall Tomorrow Classification"
)

ax.set_ylabel("Number of Records")

st.pyplot(fig)


# ---------------------------------------------------------
# TRAIN TEST SPLIT
# ---------------------------------------------------------

st.header("🤖 Machine Learning")


train = df[df["year"] < 2024].copy()
test = df[df["year"] >= 2024].copy()

if len(train) == 0 or len(test) == 0:
    st.warning(
        "The dataset does not contain both pre-2024 "
        "and 2024-or-later records."
    )
    st.stop()


X_train = train[features]
y_train = train["Rainfall_Tomorrow"]

X_test = test[features]
y_test = test["Rainfall_Tomorrow"]


# ---------------------------------------------------------
# SCALING
# ---------------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)


# ---------------------------------------------------------
# LOGISTIC REGRESSION
# ---------------------------------------------------------

logistic_model = LogisticRegression(
    max_iter=1000
)

logistic_model.fit(
    X_train_scaled,
    y_train
)

logistic_pred = logistic_model.predict(
    X_test_scaled
)


# ---------------------------------------------------------
# RANDOM FOREST
# ---------------------------------------------------------

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

rf_model.fit(
    X_train_scaled,
    y_train
)

rf_pred = rf_model.predict(
    X_test_scaled
)


# ---------------------------------------------------------
# MODEL EVALUATION FUNCTION
# ---------------------------------------------------------

def evaluate_model(y_true, prediction):

    return {
        "Accuracy": accuracy_score(
            y_true,
            prediction
        ),
        "Precision": precision_score(
            y_true,
            prediction,
            zero_division=0
        ),
        "Recall": recall_score(
            y_true,
            prediction,
            zero_division=0
        ),
        "F1 Score": f1_score(
            y_true,
            prediction,
            zero_division=0
        )
    }


logistic_results = evaluate_model(
    y_test,
    logistic_pred
)

rf_results = evaluate_model(
    y_test,
    rf_pred
)


# ---------------------------------------------------------
# MODEL COMPARISON
# ---------------------------------------------------------

results = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Random Forest"
    ],
    "Accuracy": [
        logistic_results["Accuracy"],
        rf_results["Accuracy"]
    ],
    "Precision": [
        logistic_results["Precision"],
        rf_results["Precision"]
    ],
    "Recall": [
        logistic_results["Recall"],
        rf_results["Recall"]
    ],
    "F1 Score": [
        logistic_results["F1 Score"],
        rf_results["F1 Score"]
    ]
})


st.subheader("6. Model Performance")

st.dataframe(
    results.style.format({
        "Accuracy": "{:.2%}",
        "Precision": "{:.2%}",
        "Recall": "{:.2%}",
        "F1 Score": "{:.2%}"
    }),
    use_container_width=True
)


# ---------------------------------------------------------
# MODEL PERFORMANCE CHART
# ---------------------------------------------------------

fig, ax = plt.subplots(
    figsize=(10, 5)
)

x = np.arange(len(results))
width = 0.18

metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score"
]

for i, metric in enumerate(metrics):

    ax.bar(
        x + (i - 1.5) * width,
        results[metric],
        width,
        label=metric
    )

ax.set_xticks(x)
ax.set_xticklabels(
    results["Model"]
)

ax.set_ylim(0, 1)

ax.set_ylabel("Score")
ax.set_title(
    "Model Performance Comparison"
)

ax.legend()

st.pyplot(fig)


# ---------------------------------------------------------
# CONFUSION MATRICES
# ---------------------------------------------------------

st.subheader("7. Confusion Matrices")

col1, col2 = st.columns(2)

with col1:

    st.write("Logistic Regression")

    cm = confusion_matrix(
        y_test,
        logistic_pred
    )

    fig, ax = plt.subplots(
        figsize=(5, 4)
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    st.pyplot(fig)


with col2:

    st.write("Random Forest")

    cm = confusion_matrix(
        y_test,
        rf_pred
    )

    fig, ax = plt.subplots(
        figsize=(5, 4)
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Greens",
        ax=ax
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    st.pyplot(fig)


# ---------------------------------------------------------
# RANDOM FOREST FEATURE IMPORTANCE
# ---------------------------------------------------------

st.subheader(
    "8. Random Forest Feature Importance"
)

importance = pd.DataFrame({
    "Feature": features,
    "Importance": rf_model.feature_importances_
})

importance = importance.sort_values(
    "Importance",
    ascending=False
)

fig, ax = plt.subplots(
    figsize=(10, 6)
)

ax.barh(
    importance["Feature"],
    importance["Importance"]
)

ax.invert_yaxis()

ax.set_xlabel("Importance")
ax.set_title(
    "Random Forest Feature Importance"
)

st.pyplot(fig)


# ---------------------------------------------------------
# PREDICTION SECTION
# ---------------------------------------------------------

st.header("🌦️ Predict Tomorrow's Rainfall")

st.write(
    "Enter the weather and geographical values below."
)

col1, col2, col3 = st.columns(3)

with col1:

    avg_temp = st.number_input(
        "Average Temperature",
        value=25.0
    )

    min_temp = st.number_input(
        "Minimum Temperature",
        value=20.0
    )

    max_temp = st.number_input(
        "Maximum Temperature",
        value=30.0
    )

    wind_speed = st.number_input(
        "Wind Speed",
        value=10.0
    )

with col2:

    air_pressure = st.number_input(
        "Air Pressure",
        value=1010.0
    )

    latitude = st.number_input(
        "Latitude",
        value=20.0
    )

    longitude = st.number_input(
        "Longitude",
        value=78.0
    )

with col3:

    elevation = st.number_input(
        "Elevation",
        value=100.0
    )

    month = st.slider(
        "Month",
        min_value=1,
        max_value=12,
        value=6
    )

    prediction_model = st.selectbox(
        "Prediction Model",
        [
            "Random Forest",
            "Logistic Regression"
        ]
    )


# ---------------------------------------------------------
# PREDICT
# ---------------------------------------------------------

if st.button(
    "🌧️ Predict Rainfall",
    use_container_width=True
):

    input_data = pd.DataFrame({
        "avg_temp": [avg_temp],
        "min_temp": [min_temp],
        "max_temp": [max_temp],
        "wind_speed": [wind_speed],
        "air_pressure": [air_pressure],
        "latitude": [latitude],
        "longitude": [longitude],
        "elevation": [elevation],
        "month": [month]
    })

    input_scaled = scaler.transform(
        input_data
    )

    if prediction_model == "Random Forest":

        prediction = rf_model.predict(
            input_scaled
        )[0]

        probability = rf_model.predict_proba(
            input_scaled
        )[0][1]

    else:

        prediction = logistic_model.predict(
            input_scaled
        )[0]

        probability = logistic_model.predict_proba(
            input_scaled
        )[0][1]


    st.subheader("Prediction Result")

    if prediction == 1:

        st.success(
            f"🌧️ Rainfall is predicted tomorrow. "
            f"Probability: {probability:.2%}"
        )

    else:

        st.info(
            f"☀️ No significant rainfall is predicted "
            f"tomorrow. Probability of rain: "
            f"{probability:.2%}"
        )

    # Probability chart

    probability_df = pd.DataFrame({
        "Condition": [
            "No Rain",
            "Rain"
        ],
        "Probability": [
            1 - probability,
            probability
        ]
    })

    fig, ax = plt.subplots(
        figsize=(8, 4)
    )

    ax.bar(
        probability_df["Condition"],
        probability_df["Probability"]
    )

    ax.set_ylim(0, 1)

    ax.set_ylabel("Probability")
    ax.set_title(
        "Rainfall Prediction Probability"
    )

    st.pyplot(fig)


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.markdown("---")

st.caption(
    "Rainfall Prediction System | "
    "Logistic Regression + Random Forest"
)