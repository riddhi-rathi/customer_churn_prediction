import streamlit as st
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

st.set_page_config(page_title="Customer Churn Prediction", page_icon="📊", layout="wide")
st.title("📊 Customer Churn Prediction")
st.write("Predict whether a telecom customer is likely to leave the service.")
CSV = Path(__file__).parent / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

@st.cache_resource
def train():
    df = pd.read_csv(CSV)
    d = df.drop(columns=["customerID"]).copy()
    d["TotalCharges"] = pd.to_numeric(d["TotalCharges"], errors="coerce")
    d["TotalCharges"] = d["TotalCharges"].fillna(d["TotalCharges"].mean())
    d["Churn"] = d["Churn"].map({"No": 0, "Yes": 1})
    cats = d.drop(columns=["Churn"]).select_dtypes(include=["object"]).columns
    encoded = pd.get_dummies(d, columns=cats, drop_first=True, dtype=int)
    X, y = encoded.drop(columns=["Churn"]), encoded["Churn"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.20, random_state=42, stratify=y)
    scaler = StandardScaler()
    Xtr_s, Xte_s = scaler.fit_transform(Xtr), scaler.transform(Xte)
    pca = PCA(n_components=5, random_state=42)
    Xtr_p, Xte_p = pca.fit_transform(Xtr_s), pca.transform(Xte_s)
    models = {
        "Logistic Regression (PCA)": (LogisticRegression(max_iter=1000, random_state=42), Xtr_p, Xte_p, "pca"),
        "Random Forest": (RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1, class_weight="balanced"), Xtr_s, Xte_s, "scaled")
    }
    models["Logistic Regression (PCA)"][0].fit(Xtr_p, ytr)
    models["Random Forest"][0].fit(Xtr_s, ytr)
    try:
        from xgboost import XGBClassifier
        xgb = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=.05, random_state=42, eval_metric="logloss", n_jobs=-1)
        xgb.fit(Xtr_s, ytr)
        models["XGBoost"] = (xgb, Xtr_s, Xte_s, "scaled")
    except ImportError:
        pass
    rows = []
    for name, (model, _, xtest, _) in models.items():
        pred = model.predict(xtest)
        rows.append({"Model": name, "Accuracy": accuracy_score(yte, pred), "Precision": precision_score(yte, pred, zero_division=0), "Recall": recall_score(yte, pred, zero_division=0), "F1-score": f1_score(yte, pred, zero_division=0)})
    metrics = pd.DataFrame(rows).sort_values("F1-score", ascending=False)
    best_name = metrics.iloc[0]["Model"]
    best_model, _, best_test, input_type = models[best_name]
    return X.columns, scaler, pca, best_model, best_name, input_type, metrics, confusion_matrix(yte, best_model.predict(best_test)), df

try:
    columns, scaler, pca, model, model_name, input_type, metrics, cm, df = train()
except FileNotFoundError:
    st.error("CSV not found. Put WA_Fn-UseC_-Telco-Customer-Churn.csv in the same folder as app.py.")
    st.stop()

st.subheader("Dataset overview")
a,b,c = st.columns(3)
a.metric("Total customers", f"{len(df):,}")
b.metric("Input features", len(columns))
c.metric("Churn rate", f"{(df['Churn'] == 'Yes').mean()*100:.1f}%")
st.divider()
st.subheader("Enter customer details")
with st.form("customer_form"):
    st.markdown("**Personal details**")
    a,b,c = st.columns(3)
    gender = a.selectbox("Gender", ["Female","Male"])
    senior = b.selectbox("Senior citizen", [0,1], format_func=lambda x: "Yes" if x else "No")
    partner = c.selectbox("Partner", ["Yes","No"])
    dependents = a.selectbox("Dependents", ["Yes","No"])
    tenure = b.number_input("Tenure (months)", 0, 72, 12)
    st.markdown("**Services**")
    a,b,c = st.columns(3)
    phone = a.selectbox("Phone service", ["Yes","No"])
    multiple = b.selectbox("Multiple lines", ["No","Yes","No phone service"])
    internet = c.selectbox("Internet service", ["DSL","Fiber optic","No"])
    security = a.selectbox("Online security", ["No","Yes","No internet service"])
    backup = b.selectbox("Online backup", ["No","Yes","No internet service"])
    protection = c.selectbox("Device protection", ["No","Yes","No internet service"])
    support = a.selectbox("Tech support", ["No","Yes","No internet service"])
    tv = b.selectbox("Streaming TV", ["No","Yes","No internet service"])
    movies = c.selectbox("Streaming movies", ["No","Yes","No internet service"])
    st.markdown("**Billing details**")
    a,b,c = st.columns(3)
    contract = a.selectbox("Contract", ["Month-to-month","One year","Two year"])
    paperless = b.selectbox("Paperless billing", ["Yes","No"])
    payment = c.selectbox("Payment method", ["Electronic check","Mailed check","Bank transfer (automatic)","Credit card (automatic)"])
    monthly = a.number_input("Monthly charges", min_value=0.0, value=70.0)
    total = b.number_input("Total charges", min_value=0.0, value=840.0)
    submit = st.form_submit_button("Predict churn", type="primary", use_container_width=True)

if submit:
    customer = {"gender":gender,"SeniorCitizen":senior,"Partner":partner,"Dependents":dependents,"tenure":tenure,
    "PhoneService":phone,"MultipleLines":multiple,"InternetService":internet,"OnlineSecurity":security,
    "OnlineBackup":backup,"DeviceProtection":protection,"TechSupport":support,"StreamingTV":tv,
    "StreamingMovies":movies,"Contract":contract,"PaperlessBilling":paperless,"PaymentMethod":payment,
    "MonthlyCharges":monthly,"TotalCharges":total}
    one = pd.get_dummies(pd.DataFrame([customer]), dtype=int).reindex(columns=columns, fill_value=0)
    scaled = scaler.transform(one)
    model_input = pca.transform(scaled) if input_type == "pca" else scaled
    pred = int(model.predict(model_input)[0])
    probability = float(model.predict_proba(model_input)[0][1]*100)
    st.divider()
    st.subheader("Prediction result")
    left,right = st.columns(2)
    left.metric("Estimated churn probability", f"{probability:.2f}%")
    left.progress(int(round(probability)))
    if pred:
        right.error("⚠️ This customer is predicted as **Likely to churn**.")
    else:
        right.success("✅ This customer is predicted as **Not likely to churn**.")
    st.caption(f"Model selected using test F1-score: {model_name}. A prediction is an estimate, not a guarantee.")

st.divider()
st.subheader("Model performance")
shown = metrics.copy()
for col in ["Accuracy","Precision","Recall","F1-score"]:
    shown[col] = (shown[col]*100).round(2).astype(str) + "%"
st.dataframe(shown, use_container_width=True, hide_index=True)
st.write(f"**Selected model:** {model_name}")
st.subheader("Confusion matrix")
st.dataframe(pd.DataFrame(cm, index=["Actual: No churn","Actual: Churn"], columns=["Predicted: No churn","Predicted: Churn"]), use_container_width=True)
