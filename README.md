# 🖐️ Hand Landmark Segmenter dengan OpenCV & MediaPipe  
Skrip ini menangkap gambar tangan secara real-time dari webcam, mendeteksi posisi tangan menggunakan **MediaPipe Hands**, kemudian: mengambil bounding box dari tangan (1–2 tangan sekaligus), melakukan crop dan resize ke kanvas hitam ukuran **300x300 px**, menggambar ulang **landmark tangan** dengan warna hijau dan garis koneksi merah, serta menyimpan output sebagai video `output_segmented_landmarks.mp4`.

## 📦 Dependensi  
Install terlebih dahulu:  
`pip install opencv-python mediapipe numpy`

## ▶️ Cara Menjalankan  
Jalankan dengan:  
`MediapipeKeypoint.py`  
Tekan `q` untuk keluar dari tampilan.

## 📂 Output  
- **Preview Window**: "Tangan di Tengah - Background Hitam" menampilkan hasil segmentasi real-time.  
- **Video File**: `output_segmented_landmarks.mp4` akan otomatis tersimpan berisi hasil deteksi.

## 🧠 Logika Utama  
Menggunakan webcam dengan `cv2.VideoCapture(0)`, mendeteksi landmark menggunakan `MediaPipe Hands`, menghitung bounding box seluruh tangan lalu mengubahnya menjadi kotak persegi, menggambar ulang landmark ke kanvas hitam (`np.zeros`) berukuran 300x300 px, dan menyimpan hasil ke dalam video.

## 📌 Catatan  
- `OUTPUT_SIZE = 300`: ukuran tetap hasil akhir  
- `EXTRA_PADDING = 40`: padding agar tangan tidak terpotong  
- Frame akan hitam kosong jika tidak ada tangan terdeteksi.
-----------------------------------------

# 🤖 Image Classifier with MobileNetV3Small (Keras + TensorFlow)  
Proyek ini membangun model klasifikasi gambar menggunakan transfer learning dengan arsitektur **MobileNetV3Small**. Dataset berupa gambar dari folder per kelas, kemudian diproses, dilatih, dievaluasi, dan disimpan dalam berbagai format (.h5, .keras, .tflite, .pkl).

## 📦 Dependensi  
`numpy pandas opencv-python matplotlib seaborn scikit-learn tensorflow joblib`

## 📁 Dataset  
Dataset diletakkan di folder:  
`/kaggle/input/datasetsegmen/FrameSeg5000/FrameSeg5000/`  
Setiap subfolder merepresentasikan satu kelas label.

## 📌 Proses  
1. **Load gambar & preprocessing** (resize ke 224x224, pakai `preprocess_input`)  
2. **Label encoding** dan one-hot (`to_categorical`)  
3. **Split dataset**: train (70%), val (20%), test (10%)  
4. **Model building**:  
   - Base model: `MobileNetV3Small` (imagenet weights, frozen awalnya)  
   - Custom top layer: GAP → Dense → Dropout → Dense → Output Softmax  
5. **Pretraining**: 5 epoch, hanya top-layer  
6. **Fine-tuning**: unfreeze sebagian layer (`FINE_TUNE_AT=150`), learning rate lebih kecil  
7. **EarlyStopping** dan `ModelCheckpoint` digunakan  
8. **Visualisasi akurasi & loss per epoch**  
9. **Evaluasi**: Confusion matrix & metrics (`accuracy, precision, recall, f1`)  
10. **Penyimpanan**:  
    - `.h5` dan `.keras` (model lengkap)  
    - `.tflite` (mobile deployment)  
    - `.pkl` (ekstraksi fitur latent layer)

## 📈 Output  
- Grafik akurasi & loss  
- Confusion matrix (heatmap)  
- Laporan klasifikasi (per kelas)  
- File yang disimpan:  
  - `mobilenetv3.h5`  
  - `mobilenetv3.keras`  
  - `mobilenetv3.tflite`  
  - `mobilenetv3.pkl`

## 🧠 Arsitektur Model  
- Pretrained: MobileNetV3Small (tanpa top)  
- Added layers:  
  - `GlobalAveragePooling2D → Dense(256, relu) → Dropout(0.4) → Dense(128, relu) → Dropout(0.3) → Dense(num_classes, softmax)`

## 💾 Simpan Model  
Model disimpan untuk berbagai kebutuhan: training lanjut, deploy web/app, dan ekstraksi fitur.

## 👩‍💻 Kontributor  
**Padmavati Darma Putri Tanuwijaya**  
Eksperimen segmentasi dan visualisasi tangan  
Tahun: 2025
