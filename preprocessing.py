import pandas as pd
import numpy as np
import neurokit2 as nk

# =============================================================================
# FUNGSI-FUNGSI HELPER UNTUK EKSTRAKSI FITUR
# =============================================================================

def get_acc_features(acc_raw, sampling_rate=32, window_size=5, window_shift=0.25):
    """
    Mengekstrak fitur statistik dari data Akselerometer (ACC) 3-sumbu.
    """
    # Menyiapkan list kosong untuk menampung kamus (dictionary) fitur dari setiap window
    features_list = []
    # Mengonversi ukuran dan pergeseran window dari detik ke jumlah sampel
    window_size_samples = int(window_size * sampling_rate)
    window_shift_samples = int(window_shift * sampling_rate)
    
    # Membuat iterator untuk perulangan sliding window
    windows_iterator = range(0, len(acc_raw) - window_size_samples, window_shift_samples)
    for start in windows_iterator:
        end = start + window_size_samples
        # Mengambil satu segmen/window data
        window = acc_raw[start:end]
        
        # Menetapkan timestamp pada akhir window untuk tujuan penyelarasan
        timestamp = end / sampling_rate
        features = {'timestamp': timestamp}
        
        # Menghitung fitur statistik untuk setiap sumbu (x, y, z)
        for i, axis in enumerate(['x', 'y', 'z']):
            features[f'acc_mean_{axis}'] = window[:, i].mean()
            features[f'acc_std_{axis}'] = window[:, i].std()
            features[f'acc_min_{axis}'] = window[:, i].min()
            features[f'acc_max_{axis}'] = window[:, i].max()
            
        # Menghitung magnitudo untuk merepresentasikan total pergerakan
        magnitude = np.sqrt((window**2).sum(axis=1))
        features['acc_mean_mag'] = magnitude.mean()
        features['acc_std_mag'] = magnitude.std()
        
        features_list.append(features)
        
    return pd.DataFrame(features_list)

def get_bvp_features(bvp_raw, sampling_rate=64, window_size=60, window_shift=5.0):
    """
    Mengekstrak fitur Heart Rate Variability (HRV) dari sinyal BVP mentah.
    """
    features_list = []
    window_size_samples = int(window_size * sampling_rate)
    window_shift_samples = int(window_shift * sampling_rate)
    windows_iterator = range(0, len(bvp_raw) - window_size_samples, window_shift_samples)

    for start in windows_iterator:
        end = start + window_size_samples
        window = bvp_raw[start:end].flatten()
        
        # Inisialisasi semua fitur dengan NaN di awal setiap loop
        features = {
            'timestamp': end / sampling_rate,
            'bvp_mean_hr': np.nan, 'bvp_std_hr': np.nan,
            'bvp_rmssd': np.nan, 'bvp_lf_hf_ratio': np.nan
        }

        try:
            # Memproses sinyal BVP untuk membersihkan dan mendeteksi puncak detak jantung
            ppg_signals, info = nk.ppg_process(window, sampling_rate=sampling_rate)
            peaks = info["PPG_Peaks"]
            
            # Memastikan ada cukup puncak detak jantung untuk perhitungan HRV yang valid
            if len(peaks) > 5:
                # Hitung fitur domain waktu
                hrv_time = nk.hrv_time(peaks, sampling_rate=sampling_rate, show=False)
                if "HRV_MeanNN" in hrv_time.columns and hrv_time["HRV_MeanNN"].iloc[0] > 0:
                    features['bvp_mean_hr'] = 60000 / hrv_time["HRV_MeanNN"].iloc[0]
                if "HRV_RMSSD" in hrv_time.columns:
                    features['bvp_rmssd'] = hrv_time["HRV_RMSSD"].iloc[0]
                
                # Hitung fitur domain frekuensi
                hrv_freq = nk.hrv_frequency(peaks, sampling_rate=sampling_rate, show=False)
                if 'HRV_LFHF' in hrv_freq.columns:
                    features['bvp_lf_hf_ratio'] = hrv_freq["HRV_LFHF"].iloc[0]
                    
                # Hitung standar deviasi detak jantung
                features['bvp_std_hr'] = ppg_signals["PPG_Rate"].std()
        except Exception:
            # Jika ada error apapun, biarkan fitur tetap NaN
            pass
            
        features_list.append(features)
        
    return pd.DataFrame(features_list)

def get_temp_features(temp_raw, sampling_rate=4, window_size=60, window_shift=0.25):
    """
    Mengekstrak fitur statistik dari data temperatur (TEMP).
    """
    features_list = []
    window_size_samples = int(window_size * sampling_rate)
    window_shift_samples = int(window_shift * sampling_rate)
    windows_iterator = range(0, len(temp_raw) - window_size_samples, window_shift_samples)
    
    for start in windows_iterator:
        end = start + window_size_samples
        window = temp_raw[start:end].flatten()
        
        timestamp = end / sampling_rate
        features = {'timestamp': timestamp}
        
        # Menghitung fitur statistik dasar
        features['temp_mean'] = window.mean()
        features['temp_std'] = window.std()
        features['temp_min'] = window.min()
        features['temp_max'] = window.max()
        
        # Menghitung slope (kemiringan) untuk mengetahui tren temperatur
        x_axis = np.arange(len(window))
        slope, _ = np.polyfit(x_axis, window, 1)
        features['temp_slope'] = slope
        
        features_list.append(features)

    return pd.DataFrame(features_list)

# =============================================================================
# FUNGSI UTAMA UNTUK PREPROCESSING
# =============================================================================

def preprocess_subject_data(raw_df):
    """
    Menerima DataFrame sinyal mentah dari satu subjek dan mengubahnya menjadi
    DataFrame fitur yang siap untuk model.

    Parameters:
    - raw_df (pd.DataFrame): DataFrame dengan kolom sinyal mentah 
                             (misal: 'ACC_x', 'BVP', 'TEMP').

    Returns:
    - pd.DataFrame: DataFrame berisi fitur yang telah diekstrak dan dibersihkan.
    """
    print("Memulai preprocessing...")
    
    # Ekstrak dan bersihkan sinyal mentah dari DataFrame input
    print("  - Membersihkan sinyal mentah...")
    acc_raw = raw_df[['ACC_x', 'ACC_y', 'ACC_z']].dropna().values
    bvp_raw = raw_df['BVP'].dropna().values
    temp_raw = raw_df['TEMP'].dropna().values
    
    # Ekstrak fitur dari setiap sinyal mentah
    print("  - Mengekstrak fitur ACC, BVP, TEMP...")
    df_acc = get_acc_features(acc_raw)
    # Untuk BVP, digunakan window_shift yang lebih besar agar proses lebih cepat
    df_bvp = get_bvp_features(bvp_raw, window_shift=5.0) 
    df_temp = get_temp_features(temp_raw)
    
    # Periksa jika ada kegagalan ekstraksi awal
    if df_acc.empty or df_bvp.empty or df_temp.empty:
        print("PERINGATAN: Gagal mengekstrak fitur dari salah satu sinyal. Mengembalikan DataFrame kosong.")
        return pd.DataFrame()

    # Gabungkan semua fitur berdasarkan timestamp terdekat
    print("  - Menggabungkan fitur dari semua sinyal...")
    df_merged = pd.merge_asof(df_acc, df_bvp, on='timestamp', direction='nearest')
    df_merged = pd.merge_asof(df_merged, df_temp, on='timestamp', direction='nearest')

    # Tangani nilai kosong (NaN) yang mungkin muncul
    print("  - Mengisi nilai yang hilang (NaN)...")
    df_merged.fillna(method='ffill', inplace=True) # Isi dengan nilai valid terakhir
    df_merged.fillna(method='bfill', inplace=True) # Isi sisa NaN di awal dengan nilai valid pertama
    df_merged.dropna(inplace=True) # Buang baris jika masih ada NaN
    
    print(f"Preprocessing selesai. Menghasilkan {len(df_merged)} baris fitur.")
    
    return df_merged