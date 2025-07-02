import streamlit as st
import joblib

# Load model
model = joblib.load("resume.pkl")

# UI
st.set_page_config(page_title="Resume Classifier", page_icon="📄")
st.title("📄 AI Resume Classifier")
st.markdown("Get instant prediction of resume category using Machine Learning!")

# Sidebar for example resumes
st.sidebar.title("💡 Examples")
if st.sidebar.button("Show Example for Data Science"):
    st.session_state['resume'] = "Skilled in Python, Machine Learning, Pandas, NumPy..."

if st.sidebar.button("Show Example for Web Development"):
    st.session_state['resume'] = "Experienced with HTML, CSS, JavaScript, React..."

# Input
resume_text = st.text_area("Paste your resume text below:", value=st.session_state.get('resume', ''))

if st.button("🔍 Predict"):
    if resume_text.strip() == "":
        st.warning("Please enter some text to predict.")
    else:
        prediction = model.predict([resume_text])[0]
        st.success(f"🎯 Predicted Category: **{prediction}**")
