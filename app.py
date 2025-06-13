import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import altair as alt

st.set_page_config(page_title="Rilexin", layout="wide")

# === CSS Styling agar sidebar mirip gambar ===
st.markdown("""
    <style>
    .css-1d391kg {
        background-color: #3498db !important;
    }
    .st-emotion-cache-1d391kg {
        background-color: #3498db !important;
    }
    section[data-testid="stSidebar"] .st-radio > div {
        flex-direction: column;
    }
    .stRadio > label {
        color: white !important;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# === Sidebar Navigation ===
menu = st.sidebar.radio("Navigasi", ["BERANDA", "UPLOAD & PREDIKSI", "KONTAK", "TENTANG"])

# Fungsi untuk memuat model
@st.cache_resource
def load_model_keras():
    return load_model("data/wesad_model.h5")

model = load_model_keras()

# === Konten Halaman Berdasarkan Menu ===
if menu == "BERANDA":
    st.image("data/health_img.png", width=120)
    st.markdown("<h1 style='text-align: center;'>🧠 Rilexin</h1>", unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: center;'>

    Selamat datang di aplikasi <strong>Stress Detection</strong> berbasis data sensor wearable!<br><br>

    <h3>📌 Fungsi Aplikasi</h3>
    <p style='text-align: left; display: inline-block;'>
    - Mendeteksi stres secara otomatis dari data sensor wearable.<br>
    - Menyediakan visualisasi sinyal sensor.<br>
    - Memberikan insight dari prediksi tingkat stres.<br>
    </p>

    <h3>📖 Cara Menggunakan</h3>
    <p style='text-align: left; display: inline-block;'>
    1. Pilih menu <strong>Upload & Prediksi</strong> di sidebar.<br>
    2. Unggah file CSV dari data wearable sensor Anda.<br>
    3. Pilih fitur seperti visualisasi atau prediksi stres.<br>
    4. Unduh hasil jika diperlukan.
    </p>

    </div>
    """, unsafe_allow_html=True)

elif menu == "UPLOAD & PREDIKSI":
    st.header("📁 Upload File CSV & Deteksi Stres")

    uploaded_file = st.file_uploader("Upload file CSV dari wearable sensor:", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df = df.dropna(axis=1, how='all')  # Drop kolom kosong
        st.success("File berhasil diunggah!")

        fitur = st.radio("🔧 Pilih fitur yang ingin digunakan:", ["🔍 Preview Data", "📊 Visualisasi Sinyal", "🧠 Prediksi Stres"])

        if fitur == "🔍 Preview Data":
            st.markdown("### 🔍 Preview Data")
            st.dataframe(df.head())

        elif fitur == "📊 Visualisasi Sinyal":
            st.markdown("### 📊 Visualisasi Sinyal Sensor")
            available_columns = df.columns.tolist()
            selected_sensors = st.multiselect("Pilih sensor untuk visualisasi:", available_columns, default=available_columns[:3])

            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                df = df.dropna(subset=['timestamp'])
                df = df.sort_values('timestamp')

                start_time = st.slider("Pilih waktu mulai:", min_value=df['timestamp'].min().to_pydatetime(), max_value=df['timestamp'].max().to_pydatetime(), value=df['timestamp'].min().to_pydatetime())
                end_time = st.slider("Pilih waktu akhir:", min_value=df['timestamp'].min().to_pydatetime(), max_value=df['timestamp'].max().to_pydatetime(), value=df['timestamp'].max().to_pydatetime())

                df = df[(df['timestamp'] >= start_time) & (df['timestamp'] <= end_time)]

            if selected_sensors:
                df_chart = df[selected_sensors].reset_index().melt(id_vars='index')
                chart = alt.Chart(df_chart).mark_line().encode(
                    x='index:Q',
                    y='value:Q',
                    color='variable:N'
                ).properties(width=700, height=400)
                st.altair_chart(chart, use_container_width=True)

        elif fitur == "🧠 Prediksi Stres":
            st.markdown("### 🧠 Prediksi Stres")
            if st.button("Prediksi Sekarang"):
                expected_columns = [
                    'acc_mean_x', 'acc_std_x', 'acc_min_x', 'acc_max_x',
                    'acc_mean_y', 'acc_std_y', 'acc_min_y', 'acc_max_y',
                    'acc_mean_z', 'acc_std_z', 'acc_min_z', 'acc_max_z',
                    'acc_mean_mag', 'acc_std_mag',
                    'bvp_mean_hr', 'bvp_rmssd', 'bvp_lf_hf_ratio', 'bvp_std_hr',
                    'temp_mean', 'temp_std', 'temp_min', 'temp_max', 'temp_slope'
                ]
                if all(col in df.columns for col in expected_columns):
                    scaler = StandardScaler()
                    df_scaled = scaler.fit_transform(df[expected_columns])

                    prediction = model.predict(df_scaled)
                    df['prediction'] = (prediction > 0.5).astype(int)
                    df['Hasil'] = df['prediction'].map({0: 'Tidak Stres', 1: 'Stres'})
                    st.dataframe(df['Hasil'].value_counts().reset_index().rename(columns={'index': 'Kelas', 'Hasil': 'Jumlah'}))

                    # Insight sederhana
                    if df['Hasil'].value_counts().get('Stres', 0) > df['Hasil'].value_counts().get('Tidak Stres', 0):
                        st.warning("⚠️ Mayoritas data menunjukkan kondisi stres. Perhatikan faktor lingkungan atau aktivitas.")
                    else:
                        st.success("✅ Sebagian besar data menunjukkan kondisi tidak stres. Tetap pertahankan gaya hidup sehat!")

                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button("📥 Unduh Hasil Prediksi", data=csv, file_name='hasil_prediksi_stres.csv', mime='text/csv')
                else:
                    st.error("File CSV tidak memiliki semua kolom sensor yang dibutuhkan: " + ", ".join(expected_columns))
    else:
        st.info("Silakan unggah file CSV sensor terlebih dahulu.")

elif menu == "TENTANG":
    st.title("ℹ️ Tentang")
    st.write("Aplikasi ini dikembangkan untuk deteksi stres berbasis data sensor wearable.")

elif menu == "KONTAK":
    st.title("📞 Kontak")
    st.write("Hubungi kami di mc299d5y2471@student.devacademy.id dan mc262d5y2274@student.devacademy.id.")