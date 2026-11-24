
import streamlit as st
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Customer Churn Analytics",
    page_icon="📊",
    layout="wide"
)

CSV = Path(__file__).parent / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

# --------------------------------------------------
# LIGHT / DARK MODE
# --------------------------------------------------

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

dark_mode = st.session_state.dark_mode

if dark_mode:
    background = "#111827"
    card = "#1f2937"
    text = "#f9fafb"
    muted = "#d1d5db"
    border = "#374151"
else:
    background = "#f5f7fb"
    card = "#ffffff"
    text = "#172033"
    muted = "#64748b"
    border = "#dbe2ea"

st.markdown(f"""
<style>

.stApp {{
    background-color: {background};
    color: {text};
}}

[data-testid="stHeader"] {{
    background-color: {background};
}}

[data-testid="stSidebar"] {{
    background-color: {card};
}}

/* Main text */
.stApp h1, .stApp h2, .stApp h3,
.stApp p, .stApp label,
.stApp span, .stApp div {{
    color: {text};
}}

/* Metric cards */
[data-testid="stMetric"] {{
    background-color: {card};
    padding: 20px;
    border-radius: 12px;
    border: 1px solid {border};
}}

/* IMPORTANT: Fix invisible metric numbers */
[data-testid="stMetricValue"],
[data-testid="stMetricValue"] div {{
    color: {text} !important;
    font-size: 30px !important;
    font-weight: 700 !important;
}}

[data-testid="stMetricLabel"],
[data-testid="stMetricLabel"] p {{
    color: {muted} !important;
}}

/* Forms */
[data-testid="stForm"] {{
    background-color: {card};
    padding: 20px;
    border-radius: 12px;
    border: 1px solid {border};
}}

/* Input fields */
.stTextInput input,
.stNumberInput input,
.stSelectbox div[data-baseweb="select"] {{
    color: {text} !important;
}}

/* Buttons */
.stButton button,
.stFormSubmitButton button {{
    border-radius: 9px;
    font-weight: 600;
}}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# HEADER
# --------------------------------------------------

title_col, mode_col = st.columns([12, 1])

with title_col:
    st.title("📊 Customer Churn Analytics Dashboard")

with mode_col:
    st.write("")

    if st.button(
        "☀️" if dark_mode else "🌙",
        help="Switch light/dark mode",
        use_container_width=True
    ):
        st.session_state.dark_mode = not dark_mode
        st.rerun()

st.write(
    "Understand customer behaviour, explore churn patterns "
    "and predict the likelihood of customer attrition."
)

st.divider()

# --------------------------------------------------
# MODEL TRAINING
# --------------------------------------------------

@st.cache_resource
def train():

    df = pd.read_csv(CSV)

    d = df.drop(columns=["customerID"]).copy()

    d["TotalCharges"] = pd.to_numeric(
        d["TotalCharges"], errors="coerce"
    )

    d["TotalCharges"] = d["TotalCharges"].fillna(
        d["TotalCharges"].mean()
    )

    d["Churn"] = d["Churn"].map({
        "No": 0,
        "Yes": 1
    })

    cats = d.drop(columns=["Churn"]).select_dtypes(
        include=["object"]
    ).columns

    encoded = pd.get_dummies(
        d,
        columns=cats,
        drop_first=True,
        dtype=int
    )

    X = encoded.drop(columns=["Churn"])
    y = encoded["Churn"]

    Xtr, Xte, ytr, yte = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    scaler = StandardScaler()

    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte)

    pca = PCA(
        n_components=5,
        random_state=42
    )

    Xtr_p = pca.fit_transform(Xtr_s)
    Xte_p = pca.transform(Xte_s)

    models = {}

    models["Logistic Regression (PCA)"] = (
        LogisticRegression(
            max_iter=1000,
            random_state=42
        ),
        Xtr_p,
        Xte_p,
        "pca"
    )

    models["Random Forest"] = (
        RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced"
        ),
        Xtr_s,
        Xte_s,
        "scaled"
    )

    models["Logistic Regression (PCA)"][0].fit(
        Xtr_p, ytr
    )

    models["Random Forest"][0].fit(
        Xtr_s, ytr
    )

    try:
        from xgboost import XGBClassifier

        xgb = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            random_state=42,
            eval_metric="logloss",
            n_jobs=-1
        )

        xgb.fit(Xtr_s, ytr)

        models["XGBoost"] = (
            xgb,
            Xtr_s,
            Xte_s,
            "scaled"
        )

    except ImportError:
        pass

    rows = []

    for name, (model, _, xtest, _) in models.items():

        pred = model.predict(xtest)

        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(yte, pred),
            "Precision": precision_score(
                yte, pred, zero_division=0
            ),
            "Recall": recall_score(
                yte, pred, zero_division=0
            ),
            "F1-score": f1_score(
                yte, pred, zero_division=0
            )
        })

    metrics = pd.DataFrame(rows).sort_values(
        "F1-score",
        ascending=False
    )

    best_name = metrics.iloc[0]["Model"]

    best_model, _, best_test, input_type = models[best_name]

    cm = confusion_matrix(
        yte,
        best_model.predict(best_test)
    )

    return (
        X.columns,
        scaler,
        pca,
        best_model,
        best_name,
        input_type,
        metrics,
        cm,
        df
    )


# --------------------------------------------------
# LOAD MODEL AND DATA
# --------------------------------------------------

try:

    (
        columns,
        scaler,
        pca,
        model,
        model_name,
        input_type,
        metrics,
        cm,
        df
    ) = train()

except FileNotFoundError:

    st.error(
        "Dataset not found. Keep the CSV file in the "
        "same folder as app.py."
    )

    st.stop()


# --------------------------------------------------
# DASHBOARD OVERVIEW
# --------------------------------------------------

st.header("📌 Dataset Overview")

total_customers = len(df)

churned = (df["Churn"] == "Yes").sum()

retained = (df["Churn"] == "No").sum()

churn_rate = churned / total_customers * 100

a, b, c, d = st.columns(4)

a.metric(
    "Total Customers",
    f"{total_customers:,}"
)

b.metric(
    "Customers Churned",
    f"{churned:,}"
)

c.metric(
    "Customers Retained",
    f"{retained:,}"
)

d.metric(
    "Overall Churn Rate",
    f"{churn_rate:.1f}%"
)

st.divider()


# --------------------------------------------------
# DATA VISUALIZATION
# --------------------------------------------------

st.header("📈 Customer Insights")

col1, col2 = st.columns(2)

with col1:

    st.subheader("Churn Distribution")

    churn_counts = df["Churn"].value_counts()

    churn_counts.index = [
        "Retained" if x == "No" else "Churned"
        for x in churn_counts.index
    ]

    st.bar_chart(churn_counts)

with col2:

    st.subheader("Churn by Contract Type")

    contract_churn = (
        df.groupby("Contract")["Churn"]
        .apply(lambda x: (x == "Yes").mean() * 100)
        .sort_values(ascending=False)
    )

    st.bar_chart(contract_churn)

col3, col4 = st.columns(2)

with col3:

    st.subheader("Churn by Internet Service")

    internet_churn = (
        df.groupby("InternetService")["Churn"]
        .apply(lambda x: (x == "Yes").mean() * 100)
        .sort_values(ascending=False)
    )

    st.bar_chart(internet_churn)

with col4:

    st.subheader("Churn by Customer Tenure")

    tenure_df = df.copy()

    tenure_df["Tenure Group"] = pd.cut(
        tenure_df["tenure"],
        bins=[-1, 12, 24, 48, 72],
        labels=[
            "0–12 months",
            "13–24 months",
            "25–48 months",
            "49–72 months"
        ]
    )

    tenure_churn = (
        tenure_df.groupby(
            "Tenure Group",
            observed=False
        )["Churn"]
        .apply(lambda x: (x == "Yes").mean() * 100)
    )

    st.bar_chart(tenure_churn)

st.caption(
    "The charts show churn percentages for each category, "
    "except the churn distribution chart, which shows customer counts."
)

st.divider()


# --------------------------------------------------
# CUSTOMER PREDICTION
# --------------------------------------------------

st.header("🔍 Predict Customer Churn")

st.write(
    "Enter a few important customer details to estimate "
    "whether the customer is likely to leave the service."
)

with st.form("prediction_form"):

    st.subheader("Customer Details")

    col1, col2, col3 = st.columns(3)

    with col1:

        tenure = st.number_input(
            "Tenure (months)",
            min_value=0,
            max_value=72,
            value=12
        )

        senior = st.selectbox(
            "Senior Citizen",
            ["No", "Yes"]
        )

        contract = st.selectbox(
            "Contract Type",
            [
                "Month-to-month",
                "One year",
                "Two year"
            ]
        )

    with col2:

        internet = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"]
        )

        security = st.selectbox(
            "Online Security",
            ["No", "Yes", "No internet service"]
        )

        support = st.selectbox(
            "Tech Support",
            ["No", "Yes", "No internet service"]
        )

    with col3:

        monthly = st.number_input(
            "Monthly Charges ($)",
            min_value=0.0,
            value=70.0
        )

        payment = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

        paperless = st.selectbox(
            "Paperless Billing",
            ["Yes", "No"]
        )

    submit = st.form_submit_button(
        "🔮 Predict Churn",
        type="primary",
        use_container_width=True
    )


# --------------------------------------------------
# PREDICTION RESULT
# --------------------------------------------------

if submit:

    # Default assumptions for fields not requested
    # from the user.

    customer = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": tenure,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": internet,
        "OnlineSecurity": security,
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": support,
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": monthly * tenure
    }

    customer["SeniorCitizen"] = (
        1 if senior == "Yes" else 0
    )

    one = pd.get_dummies(
        pd.DataFrame([customer]),
        dtype=int
    ).reindex(
        columns=columns,
        fill_value=0
    )

    scaled = scaler.transform(one)

    if input_type == "pca":
        model_input = pca.transform(scaled)
    else:
        model_input = scaled

    prediction = int(
        model.predict(model_input)[0]
    )

    probability = float(
        model.predict_proba(model_input)[0][1] * 100
    )

    probability = max(0, min(100, probability))

    st.divider()

    st.header("🎯 Prediction Result")

    left, right = st.columns(2)

    with left:

        st.subheader("Churn Probability")

        st.metric(
            "Estimated Risk",
            f"{probability:.1f}%"
        )

        st.progress(
            int(round(probability))
        )

        if probability < 30:
            risk = "Low Risk"
            st.success("🟢 Low Churn Risk")

        elif probability < 60:
            risk = "Moderate Risk"
            st.warning("🟡 Moderate Churn Risk")

        else:
            risk = "High Risk"
            st.error("🔴 High Churn Risk")

    with right:

        st.subheader("Prediction Indicator")

        if prediction == 1:

            st.error(
                "⚠️ Customer is predicted as likely to churn."
            )

            st.write(
                "Consider reviewing customer satisfaction, "
                "service quality and retention offers."
            )

        else:

            st.success(
                "✅ Customer is predicted as not likely to churn."
            )

            st.write(
                "The model predicts a lower likelihood "
                "of this customer leaving."
            )

    st.caption(
        f"Model selected using test F1-score: {model_name}. "
        "Risk bands are display categories; the model's "
        "predicted class uses its own classification threshold. "
        "The probability is an estimate, not a guarantee."
    )

    with st.expander("View prediction assumptions"):

        st.write(
            "The following fields were filled using default "
            "assumptions rather than individual customer input:"
        )

        st.write(
            "Gender: Female | Partner: No | Dependents: No | "
            "Phone Service: Yes | Multiple Lines: No | "
            "Online Backup: No | Device Protection: No | "
            "Streaming TV: No | Streaming Movies: No"
        )

        st.write(
            "Total Charges were estimated as Monthly Charges "
            "multiplied by Tenure."
        )


# --------------------------------------------------
# MODEL PERFORMANCE
# --------------------------------------------------

st.divider()

st.header("🤖 Model Performance")

st.write(
    "The models are compared using standard classification "
    "metrics. The model with the highest test F1-score is selected."
)

shown = metrics.copy()

for col in [
    "Accuracy",
    "Precision",
    "Recall",
    "F1-score"
]:

    shown[col] = (
        shown[col] * 100
    ).round(2).astype(str) + "%"

st.dataframe(
    shown,
    use_container_width=True,
    hide_index=True
)

st.info(
    f"Selected Model: {model_name}"
)

st.subheader("Confusion Matrix")

cm_df = pd.DataFrame(
    cm,
    index=[
        "Actual: No Churn",
        "Actual: Churn"
    ],
    columns=[
        "Predicted: No Churn",
        "Predicted: Churn"
    ]
)

st.dataframe(
    cm_df,
    use_container_width=True
)

st.caption(
    "Customer Churn Prediction | Machine Learning Project"
)
