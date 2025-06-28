import streamlit as st
import pandas as pd
import joblib

# Load model dan data
df_rank = pd.read_csv("df_future_sorted.csv")
df_perf = pd.read_csv("df_2025.csv")

ranking_model = joblib.load("ranking_pred.pkl")
performance_model = joblib.load("performance_pred.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# App
st.set_page_config(layout="centered")
st.title("Student Prediction App")

# Sidebar menu
menu = st.sidebar.radio("Pilih Jenis Prediksi", ["Prediksi Ranking", "Prediksi Student Performance"])

# Fungsi konversi input ke bentuk numerik (menggunakan encoder)
def encode_input(input_df):
    for col in input_df.columns:
        if col in label_encoder and input_df[col].dtype == 'object':
            input_df[col] = label_encoder[col].transform(input_df[col])
    return input_df

# Menu 1: Prediksi Ranking
if menu == "Prediksi Ranking":
    st.header("Prediksi Ranking Institusi")

    # Form input user (gunakan number_input untuk numerik)
    academic = st.number_input("Masukan skor reputasi akademik", min_value=0.0, max_value=100.0, step=0.1)
    employer = st.number_input("Masukan skor reputasi employer", min_value=0.0, max_value=100.0, step=0.1)
    faculty = st.number_input("Masukan skor faculty/student", min_value=0.0, max_value=100.0, step=0.1)
    sitasi = st.number_input("Masukan skor sitasi", min_value=0.0, max_value=100.0, step=0.1)
    inter_faculty = st.number_input("Masukan skor faculty internasional", min_value=0.0, max_value=100.0, step=0.1)
    inter_student = st.number_input("Masukan skor student internasional", min_value=0.0, max_value=100.0, step=0.1)

    # Buat DataFrame input model
    input_data = pd.DataFrame({
        'academic_reputation_score': [academic],
        'employer_reputation_score': [employer],
        'faculty_student_score': [faculty],
        'citations_score': [sitasi],
        'international_faculty_score': [inter_faculty],
        'international_student_score': [inter_student]
    })

    # Prediksi
    if st.button("Prediksi Ranking"):
        pred = ranking_model.predict(input_data)[0]
        st.success(f"📊 Prediksi Ranking untuk institusi tersebut: **{round(pred, 2)}**")

# Menu 2: Prediksi Student Performance
elif menu == "Prediksi Student Performance":
    st.header("Prediksi Student Performance")

    # Input skor (numerik)
    academic = st.number_input("Masukan skor reputasi akademik", min_value=0.0, max_value=100.0, step=0.1)
    faculty = st.number_input("Masukan skor faculty/student", min_value=0.0, max_value=100.0, step=0.1)
    sitasi = st.number_input("Masukan skor sitasi", min_value=0.0, max_value=100.0, step=0.1)
    inter_student = st.number_input("Masukan skor mahasiswa internasional", min_value=0.0, max_value=100.0, step=0.1)

    # Buat DataFrame input model
    input_data = pd.DataFrame({
        'academic_reputation_score': [academic],
        'faculty_student_score': [faculty],
        'citations_score': [sitasi],
        'international_student_score': [inter_student]
    })

    # Prediksi langsung (tanpa encoding)
    if st.button("Prediksi Performance"):
        pred = performance_model.predict(input_data)[0]
        st.success(f"📈 Prediksi Student Performance Score: **{round(pred, 2)}**")
