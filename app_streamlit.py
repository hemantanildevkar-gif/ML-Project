import streamlit as st
import joblib
import numpy as np

st.set_page_config(page_title="Spam Detector", page_icon="📧", layout="centered")
st.title("📧 Spam Detector ")

@st.cache_resource
def load_model():
    return joblib.load("spam_model.joblib")

model = load_model()

txt = st.text_area("Paste the email/SMS text:", height=180, placeholder="Type or paste message here...")

if st.button("Analyze"):
    if not txt.strip():
        st.warning("Please enter some text.")
    else:
        if hasattr(model, "predict_proba"):
            proba = float(model.predict_proba([txt])[0,1])
        else:
            s = float(model.decision_function([txt])[0])
            proba = 1.0 / (1.0 + np.exp(-s))
        label = "Spam" if proba >= 0.5 else "Not Spam"
        #st.metric("Spam probability", f"{proba:.2%}")
        st.success(f"Prediction: **{label}**")
        
