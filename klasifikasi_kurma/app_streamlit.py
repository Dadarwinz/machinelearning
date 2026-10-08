import streamlit as st
import pandas as pd
import joblib

model = joblib.load("model_kurma.joblib")

st.set_page_config(
	page_title = "Klasifikasi Kurma",
	page_icon="🫘"
)

st.title("🫘 Klasifikasi Kurma")
st.markdown("Aplikasi machine learning untuk klasifikasi kurma Tamr, Khalal atau Ruitab")

ukuran_cm = st.slider("Ukuran", 2.0, 6.0, 3.5)
berat_g = st.slider("Berat",3.5, 20.0, 10.0)
kekerasan_N = st.slider("Kekerasan",2.0, 90.0, 30.0)
kadar_gula_brix = st.slider("Kadar Gula", 14.0, 82.0, 40.0)
kadar_air_pct = st.slider("Kadar Air",10.0, 80.0, 40.0)
varietas = st.pills("Varietas", ["Ajwa", "Medjool","Zahidi", "Deglet Noor", "Barhi"], default="Ajwa")
tingkat_warna = st.pills("Tingkat Warna", ["Kuning", "Cokelat Muda","Cokelat Tua", "Hitam"], default="Kuning")
grade_cacat = st.pills("Grade Cacat", ["Tidak Ada", "Ringan","Sedang", "Berat"], default="Tidak Ada")

if st.button("Prediksi", type="primary"):
	data = pd.DataFrame([[ukuran_cm, berat_g, kekerasan_N, kadar_gula_brix, kadar_air_pct, varietas, tingkat_warna, grade_cacat]], columns=['ukuran_cm', 'berat_g', 'kekerasan_N', 'kadar_gula_brix',
       'kadar_air_pct', 'varietas', 'tingkat_warna', 'grade_cacat'])
	prediksi = model.predict(data)[0]
	presentase = max(model.predict_proba(data)[0])
	st.success(f"Prediksi **{prediksi}** dengan tingkat keyakinan **{presentase*100:.2f}%**")
	st.balloons()

st.divider()
st.caption("Dibuat dengan :🫘: oleh **Haidar Abqori Windraya**")