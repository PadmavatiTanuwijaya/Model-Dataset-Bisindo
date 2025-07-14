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

## 👩‍💻 Kontributor  
**Padmavati Darma Putri Tanuwijaya**  
Eksperimen segmentasi dan visualisasi tangan  
Tahun: 2025
