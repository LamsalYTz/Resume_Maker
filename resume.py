import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime

# Load model
model = joblib.load("resume.pkl")

# Load or create user data file
USER_DATA = "users.csv"
if not os.path.exists(USER_DATA):
    pd.DataFrame(columns=["email", "timestamp"]).to_csv(USER_DATA, index=False)

# Apply custom CSS
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# App title
st.title("Resume Category Classifier")
st.markdown("Predict the category of any resume instantly using AI ✨")

# --- Login Section ---
st.subheader("🔐 Login or Continue")
email = st.text_input("Enter your Email")

if email and st.button("Continue"):
    # Load user data and handle empty file
    user_df = pd.read_csv(USER_DATA)
    if user_df.empty:
        user_df = pd.DataFrame(columns=["email", "timestamp"])
    else:
        user_df = pd.read_csv(USER_DATA)
    
    # Save login info
    if email not in user_df["email"].values:
        new_row = pd.DataFrame([[email, datetime.now()]], columns=["email", "timestamp"])
        new_row.to_csv(USER_DATA, mode='a', header=False, index=False)
    st.success("✅ Login Successful")
    st.session_state.logged_in = True

# --- Prediction Section ---
if st.session_state.get("logged_in"):
    st.subheader("📄 Paste Your Resume Text")
    resume_text = st.text_area("Paste your resume text below", height=300)

    if resume_text and st.button("Predict Category"):
        pred = model.predict([resume_text])[0]
        st.success(f"🧠 Predicted Resume Category: **{pred}**")

        # Save prediction log
        log_data = pd.DataFrame([[email, resume_text[:50], pred, datetime.now()]],
                                columns=["email", "resume_preview", "prediction", "timestamp"])
        log_data.to_csv("app/prediction_log.csv", mode='a', header=not os.path.exists("app/prediction_log.csv"), index=False)

# --- Footer ---
st.markdown("---")
st.markdown("Made with ❤️ by YTz)")
