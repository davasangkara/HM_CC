import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")

num_imputer = joblib.load("num_imputer.pkl")
cat_imputer = joblib.load("cat_imputer.pkl")
cols_drop = joblib.load("cols_drop.pkl")
num_cols = joblib.load("num_cols.pkl")
cat_cols = joblib.load("cat_cols.pkl")


st.set_page_config(
    page_title="Credit Risk Prediction",
    layout="wide"
)

st.title("Credit Risk Prediction Dashboard")
st.write(
    "Aplikasi ini digunakan untuk memprediksi **probabilitas gagal bayar nasabah** "
    "menggunakan model **Logistic Regression**."
)

# ======================================================
# SIDEBAR - UPLOAD DATA
# ======================================================
st.sidebar.header("📂 Upload Dataset")
uploaded_file = st.sidebar.file_uploader(
    "Upload application_test.csv",
    type=["csv"]
)

st.header("Feature Importance")

coef = pd.Series(model.coef_[0], index=feature_columns)
coef_sorted = coef.reindex(coef.abs().sort_values(ascending=False).index)

fig, ax = plt.subplots(figsize=(9, 6))
coef_sorted.head(15).plot(kind="barh", ax=ax)
ax.set_title("Top 15 Feature Importance (Logistic Regression)")
ax.set_xlabel("Koefisien")
ax.invert_yaxis()
st.pyplot(fig)

st.markdown(
    """
**Interpretasi singkat:**
- Koefisien **positif** → meningkatkan risiko gagal bayar  
- Koefisien **negatif** → menurunkan risiko gagal bayar  
- Nilai absolut besar → pengaruh kuat
"""
)


st.header("Prediksi Risiko Kredit")

if uploaded_file is not None:
   
    data = pd.read_csv(uploaded_file)

    st.subheader("Preview Data")
    st.dataframe(data.head())

   
    data_clean = data.copy()

  
    data_clean.drop(columns=cols_drop, inplace=True, errors="ignore")


    if "DAYS_EMPLOYED" in data_clean.columns:
        data_clean["DAYS_EMPLOYED"].replace(365243, np.nan, inplace=True)

    data_clean[num_cols] = num_imputer.transform(data_clean[num_cols])
    data_clean[cat_cols] = cat_imputer.transform(data_clean[cat_cols])

    
    data_encoded = pd.get_dummies(
        data_clean,
        columns=cat_cols,
        drop_first=True
    )


    data_encoded = data_encoded.reindex(
        columns=feature_columns,
        fill_value=0
    )

    
    data_encoded.replace([np.inf, -np.inf], 0, inplace=True)
    data_encoded.fillna(0, inplace=True)

    data_scaled = scaler.transform(data_encoded)

   
    proba = model.predict_proba(data_scaled)[:, 1]

    result = pd.DataFrame({
        "SK_ID_CURR": data["SK_ID_CURR"],
        "TARGET_PROBABILITY": proba
    })

    st.subheader("Hasil Prediksi (Preview)")
    st.dataframe(result.head(20))

   
    csv = result.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇Download submission.csv",
        data=csv,
        file_name="submission.csv",
        mime="text/csv"
    )

else:
    st.info("Silakan upload file **application_test.csv** untuk memulai prediksi.")
