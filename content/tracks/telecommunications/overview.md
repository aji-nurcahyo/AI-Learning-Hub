# Track D — AI for Telecommunications: Overview, Roadmap & Studi Kasus

**Fokus Jalur:** Sistem nirkabel cerdas, analitik data jaringan (*network telemetry*), klasifikasi kualitas sinyal radio, prediksi performa tautan, dan operasi jaringan berbantuan AI.

---

## 1. Pengantar Jalur Spesialisasi & Ketepatan Metrik Radio
Jalur **AI for Telecommunications** menghubungkan teknik *Machine Learning* dan *Reinforcement Learning* dengan masalah rekayasa telekomunikasi modern. Seluruh pemodelan wajib menghormati definisi fisis metrik jaringan:
- **RSRP (*Reference Signal Received Power*, dBm):** Daya sinyal referensi rata-rata sel tujuan.
- **RSRQ (*Reference Signal Received Quality*, dB):** Kualitas relatif mempertimbangkan beban kanal (*RSSI*).
- **SNR (*Signal-to-Noise Ratio*, dB):** Perbandingan daya sinyal terhadap derau termal saja (tanpa interferensi).
- **SINR (*Signal-to-Interference-plus-Noise Ratio*, dB):** Perbandingan daya sinyal terhadap total interferensi antar-sel ditambah derau termal (**tidak boleh disamakan dengan SNR!**).
- **Latency (ms), Throughput (Mbps), Packet Loss (%):** Metrik kinerja lapisan tautan dan transportasi.

## 2. Roadmap Pembelajaran Terkurasi
1. **Tahap Fondasi:** **Chapter 06** (Probabilitas & Aljabar), **Chapter 07** (Python Stack), **Chapter 08** (Pembersihan & EDA Data).
2. **Tahap Inti Analitik Jaringan (ML):** **Chapter 11–19** (Workflow bebas leakage, Regresi performa, Klasifikasi sinyal, Ansambel, SVM/KNN, Clustering anomali, Reduksi dimensi PCA, serta Evaluasi & Explainability).
3. **Tahap Lanjutan Telekomunikasi Cerdas:** **Chapter 27** (NLP log insiden jaringan), **Chapter 35** (RL Alokasi Spektrum/Daya simulasi), **Chapter 36** (MLOps & Drift), **Chapter 37** (Edge AI di node jaringan), **Chapter 39** (AI Safety & Tata Kelola).

## 3. Studi Kasus Domain Telekomunikasi

### Studi Kasus 1: Deteksi Dini Degradasi Kualitas Tautan Nirkabel
- **Masalah:** Sel dengan `RSRP` tampak cukup kuat (`-90 dBm`) tetap mengalami penurunan *Throughput* drastis akibat interferensi tetangga (`SINR = 4 dB` jauh lebih rendah dari `SNR = 15 dB`).
- **Solusi:** `wireless_quality_lab.py` memisahkan tegas fitur `SNR` dan `SINR`, mengalahkan *Majority Baseline* (`81.50%`) memakai `RandomForest` (`98.50%`, Macro-F1 `0.9749`). **Batasan:** Menggunakan dataset sintetis—tidak mengklaim hasil uji lapangan jaringan seluler operator tanpa validasi *drive-test*.

### Studi Kasus 2: Deteksi Pergeseran Pola Trafik Jam Sibuk (*Concept Drift*)
- **Masalah:** Pergeseran profil pengguna (misalnya acara besar atau gangguan menara) mengubah hubungan fitur radio vs kehilangan paket.
- **Solusi:** Pemantauan kontinu memakai PSI dan KS Test (`Chapter 36`) sebelum mengaktifkan ulang pelatihan model dengan persetujuan insinyur jaringan.

