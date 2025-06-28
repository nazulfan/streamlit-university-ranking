import streamlit as st
import pandas as pd
import plotly.express as px

st.write("Selamat datang di Streamlit")

name = st.text_input("Masukan Nama Anda")
if name:
    st.write("holla", name)

data = {"usia": [20,34,22,40,32,28,37],
        "tinggi": [180,167,178,177,165,170,183],
        "gender": ["F","F","M","F","M","M","M"]}

df = pd.DataFrame(data)

st.title("Tabel Usia & Tinggi")
st.header("Contoh tabel")
st.dataframe(df)

st.subheader("Visualisasi Pie Chart")

pie_chart = px.pie(df, names="gender", values="usia", title="Distribusi Usia terhadap Gender")
st.plotly_chart(pie_chart)