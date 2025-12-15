import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")
cols_drop = joblib.load("cols_drop.pkl")

st.set_page_config(page_title="Credit Risk Prediction", layout="wide")

st.title("Credit Risk Prediction Dashboard")
st.write("Aplikasi prediksi probabilitas gagal bayar menggunakan Logistic Regression")

coef = pd.Series(model.coef_[0], index=feature_columns)
coef_sorted = coef.reindex(coef.abs().sort_values(ascending=False).index)

fig, ax = plt.subplots(figsize=(9, 6))
coef_sorted.head(15).plot(kind="barh", ax=ax)
ax.set_title("Top 15 Feature Importance")
ax.set_xlabel("Koefisien")
ax.invert_yaxis()
st.pyplot(fig)

st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload application_test.csv", type=["csv"])

st.header("Prediksi Risiko Kredit")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.subheader("Preview Data")
    st.dataframe(data.head())

    data_clean = data.copy()
    data_clean.drop(columns=cols_drop, inplace=True, errors="ignore")

    if "DAYS_EMPLOYED" in data_clean.columns:
        data_clean["DAYS_EMPLOYED"] = data_clean["DAYS_EMPLOYED"].replace(365243, 0)

    data_encoded = pd.get_dummies(data_clean, drop_first=True)
    data_encoded = data_encoded.reindex(columns=feature_columns, fill_value=0)
    data_encoded.replace([np.inf, -np.inf], 0, inplace=True)
    data_encoded.fillna(0, inplace=True)

    data_scaled = scaler.transform(data_encoded.values)
    proba = model.predict_proba(data_scaled)[:, 1]

    result = pd.DataFrame({
        "SK_ID_CURR": data["SK_ID_CURR"],
        "TARGET": proba
    })

    st.subheader("Hasil Prediksi")
    st.dataframe(result.head(20))

    csv = result.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download submission.csv",
        data=csv,
        file_name="submission.csv",
        mime="text/csv"
    )
else:
    st.info("Silakan upload file application_test.csv")
