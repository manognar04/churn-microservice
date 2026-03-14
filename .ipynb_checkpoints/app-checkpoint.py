import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(page_title="Telecom Network Health Dashboard", layout="wide")

st.title("📊 Telecom Network Health Dashboard")

# ----------------------------
# LOAD DATA
# ----------------------------
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Basic cleaning
if "customerID" in df.columns:
    df.drop("customerID", axis=1, inplace=True)

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.dropna(inplace=True)

# ----------------------------
# LOAD MODEL
# ----------------------------
model = joblib.load("churn_model.pkl")

# ----------------------------
# KPI SECTION
# ----------------------------
st.subheader("📌 Key Metrics")

col1, col2, col3 = st.columns(3)

col1.metric("Total Customers", len(df))
col2.metric("Avg Monthly Charges", round(df["MonthlyCharges"].mean(), 2))
col3.metric("Avg Tenure", round(df["tenure"].mean(), 2))

# ----------------------------
# CHURN PREDICTION SECTION
# ----------------------------
st.subheader("🔮 Predict Customer Churn")

tenure = st.number_input("Tenure (Months)", 0, 100, 12)
monthly = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
total = st.number_input("Total Charges", 0.0, 10000.0, 1000.0)

if st.button("Predict Churn"):

    # Create temporary dataframe for prediction
    input_df = pd.DataFrame({
        "tenure": [tenure],
        "MonthlyCharges": [monthly],
        "TotalCharges": [total]
    })

    # Align with training features
    full_df = pd.get_dummies(df, drop_first=True)
    X_cols = full_df.drop("Churn", axis=1).columns

    temp = pd.DataFrame(columns=X_cols)
    temp.loc[0] = 0

    for col in input_df.columns:
        if col in temp.columns:
            temp[col] = input_df[col]

    prediction = model.predict(temp)

    if prediction[0] == 1:
        st.error("⚠ Customer is likely to CHURN")
    else:
        st.success("✅ Customer is NOT likely to churn")

# ----------------------------
# CLUSTERING SECTION
# ----------------------------
st.subheader("📈 Customer Segmentation (K-Means)")

features = ["tenure", "MonthlyCharges", "TotalCharges"]
X_cluster = df[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

kmeans = KMeans(n_clusters=3, random_state=42)
df["Cluster"] = kmeans.fit_predict(X_scaled)

fig, ax = plt.subplots()
scatter = ax.scatter(df["MonthlyCharges"], df["TotalCharges"], c=df["Cluster"])
ax.set_xlabel("Monthly Charges")
ax.set_ylabel("Total Charges")
ax.set_title("Customer Clusters")

st.pyplot(fig)

# ----------------------------
# ANOMALY DETECTION
# ----------------------------
st.subheader("🚨 Anomaly Detection")

distances = kmeans.transform(X_scaled)
min_dist = np.min(distances, axis=1)

threshold = np.percentile(min_dist, 95)
df["Anomaly"] = min_dist > threshold

anomaly_count = df["Anomaly"].sum()

st.metric("Detected Anomalies (Top 5%)", anomaly_count)

st.write("Sample Anomalies:")
st.dataframe(df[df["Anomaly"] == True].head())