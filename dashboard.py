import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
from sklearn.preprocessing import LabelEncoder
import numpy as np

# ------------------------
# Load data dan model
# ------------------------
df = pd.read_csv("customers.csv", sep=';')

# Load model dan encoder jika tersedia
try:
    model = joblib.load("bonus_model.pkl")
    le_job = joblib.load("jobtitle_encoder.pkl")
except:
    model = None
    le_job = None

# ------------------------
# Sidebar: Menu Navigasi
# ------------------------
menu = st.sidebar.radio("Pilih Menu", ["Dashboard", "Predict Bonus"])

# ------------------------
# MENU 1: DASHBOARD
# ------------------------
if menu == "Dashboard":
    st.title("Dashboard Customers Analytics")

    # Sidebar filters
    st.sidebar.header("Filter Data")

    departments = df['Department'].dropna().unique()
    selected_dept = st.sidebar.selectbox("Pilih Department", departments)

    genders = df['Gender'].dropna().unique()
    selected_gender = st.sidebar.multiselect("Pilih Gender", genders, default=list(genders))

    min_age = int(df['Age'].min())
    max_age = int(df['Age'].max())
    selected_age = st.sidebar.slider("Pilih Rentang Usia", min_value=min_age, max_value=max_age, value=(min_age, max_age))

    # Terapkan filter ke data
    filtered_df = df[
        (df['Department'] == selected_dept) &
        (df['Gender'].isin(selected_gender)) &
        (df['Age'] >= selected_age[0]) &
        (df['Age'] <= selected_age[1])
    ]

    st.dataframe(filtered_df)

    st.header("Visualisasi")
    col1, col2 = st.columns(2)

    with col1:
        grouped = filtered_df.groupby(['Department', 'Gender']).size().reset_index(name='Count')
        if not grouped.empty:
            fig = px.pie(grouped, names='Gender', values='Count',
                         title=f'Distribusi Gender di Department {selected_dept}')
            st.plotly_chart(fig)
        else:
            st.warning("Tidak ada data yang cocok dengan filter.")

    with col2:
        avg_salary = df[
            (df['Gender'].isin(selected_gender)) &
            (df['Age'] >= selected_age[0]) &
            (df['Age'] <= selected_age[1])
        ].groupby('Department')['AnnualSalary'].mean().reset_index()

        fig2 = px.bar(avg_salary, x='Department', y='AnnualSalary',
                      title="Rata-Rata Gaji per Department", labels={'AnnualSalary': 'Rata-Rata Gaji'})
        st.plotly_chart(fig2)

    st.header("Visualisasi Lanjutan (Analisis Tajam)")
    col1, col2, col3 = st.columns(3)

    with col1:
        salary_by_gender = filtered_df.groupby('Gender')['AnnualSalary'].sum().reset_index()
        if not salary_by_gender.empty:
            fig_pie2 = px.pie(salary_by_gender, names='Gender', values='AnnualSalary', title='Proporsi Total Gaji per Gender')
            st.plotly_chart(fig_pie2)

    with col2:
        count_by_age = filtered_df['Age'].value_counts().sort_index().reset_index()
        count_by_age.columns = ['Age', 'Count']
        if not count_by_age.empty:
            fig_line2 = px.line(count_by_age, x='Age', y='Count', title='Jumlah Customer per Usia', markers=True)
            st.plotly_chart(fig_line2)

    with col3:
        avg_age_dept = filtered_df.groupby('Department')['Age'].mean().reset_index()
        if not avg_age_dept.empty:
            fig_bar2 = px.bar(avg_age_dept, x='Department', y='Age', title='Rata-Rata Usia per Department', labels={'Age': 'Rata-Rata Usia'})
            st.plotly_chart(fig_bar2)

# ------------------------
# MENU 2: PREDICT BONUS
# ------------------------
elif menu == "Predict Bonus":
    st.title("Prediksi Bonus Karyawan")

    if model is None or le_job is None:
        st.error("Model atau LabelEncoder belum tersedia. Silakan latih dan simpan terlebih dahulu.")
    else:
        age = st.number_input("Usia", min_value=18, max_value=100)
        salary = st.number_input("Gaji Tahunan", min_value=0)
        job_titles = df['JobTitle'].dropna().unique()
        job_input = st.selectbox("Job Title", sorted(job_titles))

        if st.button("Prediksi Bonus"):
            # Encode job title
            job_encoded = le_job.transform([job_input])[0]

            # Prediksi
            X_pred = np.array([[age, salary, job_encoded]])
            pred_bonus = model.predict(X_pred)[0]

            st.subheader("Hasil Prediksi Bonus")
            st.success(f"Prediksi Bonus: {pred_bonus * 100:,.2f} %")

