import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import joblib
from tensorflow.keras.models import load_model

# Impor fungsi preprocessing dari preprocessing.py
import preprocessing

# ===================================================================
# PENGATURAN HALAMAN & CACHING
# ===================================================================

st.set_page_config(page_title="Rilexin - Deteksi Stres", layout="wide", initial_sidebar_state="expanded")

# Fungsi untuk memuat model dan scaler dengan cache agar tidak di-load berulang kali
@st.cache_resource
def load_model_keras():
    """Memuat model TensorFlow dari file .h5"""
    try:
        return load_model("data/wesad_model.h5")
    except Exception as e:
        st.error(f"Error memuat model: {e}")
        return None

@st.cache_resource
def load_scaler():
    """Memuat objek StandardScaler dari file .pkl"""
    try:
        return joblib.load("data/scaler.pkl")
    except Exception as e:
        st.error(f"Error memuat scaler: {e}")
        return None

# Memuat model dan scaler di awal
model = load_model_keras()
scaler = load_scaler()

# Mendapatkan nama fitur yang diharapkan oleh model dari scaler
if scaler:
    try:
        EXPECTED_FEATURES = scaler.feature_names_in_
    except AttributeError:
        # Fallback jika scaler tidak memiliki feature_names_in_
        st.warning("Tidak dapat mengambil nama fitur dari scaler. Pastikan data sesuai.")
        EXPECTED_FEATURES = None
else:
    EXPECTED_FEATURES = None


# ===================================================================
# TAMPILAN HALAMAN
# ===================================================================

def display_beranda():
    """Menampilkan konten halaman Beranda."""
    st.image("data/health_img.png", width=120)
    st.markdown("<h1 style='text-align: center; color: #2C3E50;'>🧠 Rilexin</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 18px;'>Mendeteksi Stres, Menemukan Ketenangan.</p>", unsafe_allow_html=True)
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 📌 Fungsi Aplikasi")
        st.info("""
        - **Deteksi Otomatis:** Menganalisis data sensor untuk mendeteksi tingkat stres.
        - **Visualisasi Data:** Menampilkan sinyal sensor mentah dalam grafik interaktif.
        - **Insight Cepat:** Memberikan ringkasan hasil prediksi yang mudah dipahami.
        """)
    with col2:
        st.markdown("### 📖 Cara Menggunakan")
        st.warning("""
        1. **Buka Menu:** Pilih `UPLOAD & PREDIKSI` di sidebar.
        2. **Unggah File:** Upload file data mentah Anda dalam format `.csv`.
        3. **Tunggu Proses:** Biarkan aplikasi mengekstrak fitur secara otomatis.
        4. **Lihat Hasil:** Pilih aksi untuk melihat preview, visualisasi, atau hasil prediksi.
        """)

def display_upload_prediksi():
    """Menampilkan konten halaman Upload & Prediksi dengan alur kerja baru."""
    st.header("📁 Unggah & Prediksi")
    
    uploaded_file = st.file_uploader(
        "Pilih file CSV data sensor mentah Anda:",
        type=["csv"],
        help="Pastikan file CSV memiliki kolom: ACC_x, ACC_y, ACC_z, BVP, TEMP."
    )

    if uploaded_file is not None:
        if st.session_state.get('file_name') != uploaded_file.name:
            # Gunakan spinner untuk menandakan proses yang sedang berjalan
            with st.spinner("Membaca dan memproses file mentah... Ini mungkin memakan waktu beberapa saat."):
                # 1. Baca file
                st.session_state.raw_df = pd.read_csv(uploaded_file)
                st.session_state.raw_df.drop(columns=['Unnamed: 0'], inplace=True, errors='ignore')
                st.session_state.file_name = uploaded_file.name
                
                # 2. Lakukan preprocessing/ekstraksi fitur
                feature_df = preprocessing.preprocess_subject_data(st.session_state.raw_df)
                
                # 3. Simpan hasil atau tampilkan error
                if feature_df is None or feature_df.empty:
                    st.error("Gagal memproses file. Pastikan format data benar dan tidak kosong.")
                    # Bersihkan state jika gagal
                    for key in list(st.session_state.keys()):
                        del st.session_state[key]
                else:
                    # Simpan data yang berhasil diproses ke session state
                    st.session_state.feature_df = feature_df
                    st.success(f"File '{uploaded_file.name}' berhasil diproses! Silakan pilih aksi di bawah.")

    # Tampilkan opsi hanya jika data fitur SUDAH SIAP di session_state
    if 'feature_df' in st.session_state:
        df_raw = st.session_state.raw_df
        feature_df = st.session_state.feature_df
        
        fitur = st.radio(
            "🔧 Pilih aksi yang ingin dilakukan:",
            ["🧠 Hasil Prediksi", "📊 Visualisasi Sinyal", "🔍 Preview Data Mentah", "🔬 Preview Data Fitur"],
            horizontal=True
        )
        st.markdown("---")

        if fitur == "🧠 Hasil Prediksi":
            st.markdown("#### Hasil Deteksi Stres")
            # Cek kesiapan model dan scaler sekali lagi
            if model is None or scaler is None or EXPECTED_FEATURES is None:
                st.error("Model atau Scaler gagal dimuat. Proses tidak bisa dilanjutkan.")
            else:
                with st.spinner("Menyiapkan hasil..."):
                    # 1. SCALING
                    X_to_predict = feature_df[EXPECTED_FEATURES]
                    X_scaled = scaler.transform(X_to_predict)
                    
                    # 2. PREDIKSI
                    predictions_proba = model.predict(X_scaled)
                    predictions = (predictions_proba > 0.5).astype(int).flatten()

                    # 3. Tampilkan Hasil
                    display_prediction_results(predictions, feature_df['timestamp'])

        elif fitur == "📊 Visualisasi Sinyal":
            display_signal_visualization(df_raw)

        elif fitur == "🔍 Preview Data Mentah":
            st.markdown("#### Tampilan 5 baris pertama data mentah:")
            st.dataframe(df_raw.head())
            st.markdown(f"Dimensi data mentah: **{df_raw.shape[0]} baris** x **{df_raw.shape[1]} kolom**")
            
        elif fitur == "🔬 Preview Data Fitur":
            st.markdown("#### Tampilan 5 baris pertama data fitur (setelah preprocessing):")
            st.dataframe(feature_df.head())
            st.markdown(f"Dimensi data fitur: **{feature_df.shape[0]} baris** x **{feature_df.shape[1]} kolom**")

def display_prediction_results(predictions, timestamps):
    """Fungsi terpisah untuk menampilkan semua hasil prediksi."""
    col1, col2 = st.columns([1, 2])
    with col1:
        stress_count = np.sum(predictions)
        non_stress_count = len(predictions) - stress_count
        stress_percentage = (stress_count / len(predictions)) * 100 if len(predictions) > 0 else 0
        st.metric(label="Total Jendela Waktu", value=f"{len(predictions)}")
        st.metric(label="Terdeteksi Stres", value=f"{stress_count} ({stress_percentage:.1f}%)", delta_color="inverse")
        if stress_percentage > 50: st.error("Tingkat stres terdeteksi tinggi.")
        else: st.success("Tingkat stres tampak terkendali.")

    with col2:
        df_pie = pd.DataFrame({'Status': ['Stres', 'Tidak Stres'], 'Jumlah': [stress_count, non_stress_count]})
        pie_chart = alt.Chart(df_pie).mark_arc(innerRadius=50).encode(
            theta=alt.Theta(field="Jumlah", type="quantitative"),
            color=alt.Color(field="Status", type="nominal", scale=alt.Scale(domain=['Stres', 'Tidak Stres'], range=['#E74C3C', '#2ECC71']))
        ).properties(title="Distribusi Hasil Prediksi")
        st.altair_chart(pie_chart, use_container_width=True)

    st.markdown("---")
    st.markdown("#### 🕒 Garis Waktu Deteksi Stres")
    results_df = pd.DataFrame({'timestamp_sec': timestamps, 'prediksi': predictions})
    stress_events = results_df[results_df['prediksi'] == 1]

    if not stress_events.empty:
        st.write("Grafik di bawah ini menandai titik-titik waktu (dalam detik) di mana stres terdeteksi.")
        timeline_chart = alt.Chart(stress_events).mark_tick(
            color='red', thickness=2, size=20
        ).encode(
            x=alt.X('timestamp_sec', title='Garis Waktu (detik)'),
            tooltip=[alt.Tooltip('timestamp_sec', title='Stres terdeteksi pada detik ke')]
        ).properties(
            title='Kejadian Stres Terdeteksi Selama Perekaman Data',
            height=400
            )
        st.altair_chart(timeline_chart, use_container_width=True)
    else:
        st.success("Luar biasa! Tidak ada periode stres yang signifikan terdeteksi pada data Anda.")

def display_signal_visualization(df_raw):
    """Fungsi terpisah untuk menampilkan visualisasi sinyal."""
    st.markdown("#### Visualisasi Sinyal Sensor Mentah")
    available_cols = df_raw.select_dtypes(include=np.number).columns.tolist()
    selected_sensors = st.multiselect(
        "Pilih sinyal untuk divisualisasikan:", 
        available_cols, 
        default=available_cols[:3] if len(available_cols) >= 3 else available_cols
    )
    if selected_sensors:
        charts = []
        for col in selected_sensors:
            df_plot = df_raw[[col]].dropna().reset_index()
            chart = alt.Chart(df_plot).mark_line().encode(
                x=alt.X('index', title='Sampel'),
                y=alt.Y(col, title='Amplitudo'),
                tooltip=[col]
            ).properties(title=f'Sinyal {col}', width='container', height=200)
            charts.append(chart)
        st.altair_chart(alt.vconcat(*charts), use_container_width=True)

def display_tentang():
    """Menampilkan konten halaman Tentang."""
    st.title("ℹ️ Tentang Rilexin")
    st.info("""
    Aplikasi ini adalah prototipe yang dikembangkan untuk mendeteksi stres menggunakan data dari sensor wearable. 
    Dengan memanfaatkan machine learning, Rilexin bertujuan untuk memberikan insight awal mengenai kondisi fisiologis pengguna 
    yang berkaitan dengan stres.
    """)

def display_kontak():
    """Menampilkan konten halaman Kontak."""
    st.title("📞 Kontak")
    st.write("Untuk pertanyaan atau masukan, silakan hubungi tim kami melalui email:")
    st.code("mc299d5y2471@student.devacademy.id")
    st.code("mc262d5y2274@student.devacademy.id")


# ===================================================================
# NAVIGASI UTAMA APLIKASI
# ===================================================================

st.sidebar.title("Navigasi")
menu = st.sidebar.radio(
    "Pilih Halaman:",
    ["BERANDA", "UPLOAD & PREDIKSI", "TENTANG", "KONTAK"],
    label_visibility="collapsed"
)

# Menampilkan halaman sesuai pilihan menu
if menu == "BERANDA":
    display_beranda()
elif menu == "UPLOAD & PREDIKSI":
    display_upload_prediksi()
elif menu == "TENTANG":
    display_tentang()
elif menu == "KONTAK":
    display_kontak()