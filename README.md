# Laporan Proyek Machine Learning - Reynaldo Arya Budi Trisna


## Domain Proyek
  **Latar Belakang**:
  Masalah prediksi attrition karyawan (karyawan yang meninggalkan perusahaan) menjadi salah satu tantangan utama dalam manajemen SDM.
  Dengan memahami faktor-faktor yang mempengaruhi keputusan karyawan untuk meninggalkan perusahaan, perusahaan dapat mengambil langkah-langkah preventif untuk meningkatkan retensi karyawan.
  Dataset ini berisi informasi tentang karyawan dan status attrition mereka, yang bisa membantu perusahaan dalam merancang kebijakan yang lebih baik terkait dengan kepuasan karyawan dan pengelolaan tenaga kerja.
  
  **Rubrik/Kriteria Tambahan (Opsional)**:
  Masalah ini penting karena tingginya tingkat pergantian karyawan dapat menambah biaya operasional perusahaan, seperti biaya rekrutmen dan pelatihan.
  Oleh karena itu, memprediksi attrition dengan akurat memungkinkan perusahaan untuk melakukan tindakan yang lebih efektif, seperti program pelatihan yang lebih baik, peningkatan kesejahteraan, dan pengelolaan beban kerja.

  **Referensi data**: https://www.kaggle.com/code/janiobachmann/attrition-in-an-organization-why-workers-quit/

## Business Understanding 
**Problem Statement** :
  - Perusahaan ingin melakukan proses identifikasi cepat menggunakan model machine learning / deep learning untuk mempermudah perusahaan terkait analisa attrition karyawan.
  - Perusahaan ingin memprediksi attrition karyawan berdasarkan berbagai faktor yang mempengaruhi keputusan mereka untuk tetap bekerja atau meninggalkan perusahaan.
 
**Goals**:
  - Membangun model prediksi attrition karyawan untuk memprediksi apakah karyawan akan meninggalkan perusahaan.
  - Mengidentifikasi faktor-faktor yang paling signifikan yang mempengaruhi attrition.
  
**Solution Statement** :
  - Solution 1: Menggunakan Artificial Neural Network (ANN) untuk meningkatkan akurasi prediksi dan menangani masalah hubungan kompleks antar variabel.
  - Solution 2: Menggunakan Random Forest Tree untuk mengetahui faktor dari Attrtion yang paling penting
 
## Data Understanding 
  - Dataset : 
      - Referensi data : https://www.kaggle.com/code/janiobachmann/attrition-in-an-organization-why-workers-quit/
  - Informasi Dataset
    Dataset ini berisi informasi mengenai 1.470 karyawan dengan kolom yang mencakup faktor-faktor yang dapat mempengaruhi attrition, seperti:
    ```
    Age: Usia karyawan
    DistanceFromHome: Jarak tempat tinggal karyawan dari kantor
    Education: Tingkat pendidikan
    JobRole: Peran pekerjaan karyawan
    MonthlyIncome: Gaji bulanan
    Attrition: Apakah karyawan meninggalkan perusahaan (1 = Yes, 0 = No)
    ```
  - Tahapan :
    - EDA dan Visualisasi : memahami distribusi target (attrition), serta hubungan antara variabel independen seperti monthly income dan attrition.
      - ![image](https://github.com/user-attachments/assets/5f16ce66-365f-44ef-af5b-e3292b97de08)
      
        Insight : Karyawan dengan gaji bulanan yang lebih rendah cenderung lebih mungkin meninggalkan perusahaan, meskipun ada beberapa kasus di mana karyawan dengan gaji tinggi juga memilih untuk pergi.
      - ![image](https://github.com/user-attachments/assets/0a555505-9bfd-4bae-8ef3-3de377e97cc2)
     
        Insight : Karyawan yang lebih muda (dalam rentang usia 20-an hingga awal 30-an) cenderung lebih mungkin untuk meninggalkan perusahaan, sementara mereka yang berusia lebih matang (sekitar usia 40-an) lebih   cenderung untuk tetap tinggal.
      - ![image](https://github.com/user-attachments/assets/8094ee47-85c8-4faf-abb0-a87369a00433)
     
        Insight : Karyawan dengan pengalaman kerja yang lebih singkat cenderung lebih mungkin meninggalkan perusahaan. Hal ini dapat menunjukkan bahwa karyawan dengan pengalaman lebih sedikit mungkin merasa tidak puas atau mencari peluang lain lebih cepat.
     
 ## Data Preparation
   - Pemrosesan Data :
       - One-Hot Encoding untuk variabel kategorikal seperti JobRole, Department, dan Attrition.
       - Pembagian data menjadi data pelatihan dan pengujian dengan rasio 70:30.
       - Normalisasi data numerik menggunakan StandardScaler untuk memudahkan pelatihan model dan meningkatkan kinerja model deep learning.
  - Alasan : Tahapan data preparation penting untuk memastikan bahwa model dapat belajar dengan efektif. One-Hot Encoding memastikan bahwa data kategorikal dapat diproses oleh model machine learning,
    sementara normalisasi diperlukan untuk menghindari bias akibat perbedaan skala antar fitur.

## Modelling
   - Algoritma :
     ```Artificial Neural Network (ANN): Model deep learning yang lebih kompleks dan lebih baik dalam memodelkan hubungan non-linear antara variabel.```
   - Parameter ANN :
     ```
     Input Layer: Jumlah neuron sama dengan jumlah fitur input.
     Hidden Layer: 3 layer dengan neuron masing-masing 128, 64, dan 32, dengan fungsi aktivasi ReLU.
     Dropout Layer: Untuk menghindari overfitting, dropout layer ditambahkan dengan tingkat dropout 0.5.
     Output Layer: Menggunakan sigmoid untuk klasifikasi biner.
     ```
  - Kelebihan dan Kekurangan Model :
    ANN: Kelebihan - dapat menangani hubungan non-linear dengan lebih baik. Kekurangan - membutuhkan waktu pelatihan lebih lama dan rentan terhadap overfitting.

  - Proses Improvement :
    Untuk model ANN, dilakukan hyperparameter tuning dan early stopping untuk mencegah overfitting dan meningkatkan akurasi.
    
## Evaluation
   - Metriks :
       - Accuracy: Persentase prediksi yang benar dari total prediksi.
       - Precision: Proporsi prediksi positif yang benar.
       - Recall: Proporsi karyawan yang benar-benar meninggalkan perusahaan dan diprediksi dengan benar.
       - F1-Score: Rata-rata harmonis antara precision dan recall.
   - Hasil Evaluasi : ```Model deep learning (ANN) memberikan hasil evaluasi yang baik dengan accuracy sekitar 82% pada data validasi. F1-Score digunakan untuk menilai keseimbangan antara precision dan recall.```

**---Ini adalah bagian akhir laporan---**
