#Proyek Prediksi Attrition Karyawan dengan Deep Learning

1. Domain Proyek
- Latar Belakang :
  Masalah prediksi attrition karyawan (karyawan yang meninggalkan perusahaan) menjadi salah satu tantangan utama dalam manajemen SDM.
  Dengan memahami faktor-faktor yang mempengaruhi keputusan karyawan untuk meninggalkan perusahaan, perusahaan dapat mengambil langkah-langkah preventif untuk meningkatkan retensi karyawan.
  Dataset ini berisi informasi tentang karyawan dan status attrition mereka, yang bisa membantu perusahaan dalam merancang kebijakan yang lebih baik terkait dengan kepuasan karyawan dan pengelolaan tenaga kerja.
  Masalah ini penting karena tingginya tingkat pergantian karyawan dapat menambah biaya operasional perusahaan, seperti biaya rekrutmen dan pelatihan.
  Oleh karena itu, memprediksi attrition dengan akurat memungkinkan perusahaan untuk melakukan tindakan yang lebih efektif, seperti program pelatihan yang lebih baik, peningkatan kesejahteraan, dan pengelolaan beban kerja.

2. Business Understanding 
- Problem Statement :
  Perusahaan ingin memprediksi attrition karyawan berdasarkan berbagai faktor yang mempengaruhi keputusan mereka untuk tetap bekerja atau meninggalkan perusahaan.
  Model ini akan membantu manajer HR untuk mengambil keputusan yang lebih baik dalam mempertahankan karyawan berpotensi.
- Goals :
  - Membangun model prediksi attrition karyawan untuk memprediksi apakah karyawan akan meninggalkan perusahaan.
  - Mengidentifikasi faktor-faktor yang paling signifikan yang mempengaruhi attrition.
- Solution Statement :
  - Solution 1: Menggunakan Logistic Regression sebagai baseline model untuk prediksi attrition. Model ini dapat memberikan pemahaman awal mengenai faktor-faktor yang mempengaruhi attrition.
  - Solution 2: Menggunakan Random Forest Classifier atau Artificial Neural Network (ANN) untuk meningkatkan akurasi prediksi dan menangani masalah hubungan kompleks antar variabel.
 
3. Data Understanding
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
  - Sumber Dataset : https://www.kaggle.com/code/janiobachmann/attrition-in-an-organization-why-workers-quit/input
  - Tahapan :
    - EDA : memahami distribusi target (attrition), serta hubungan antara variabel independen seperti monthly income dan attrition.
    - Visualisasi : menggali wawasan lebih dalam tentang distribusi variabel dan korelasi antar fitur.
      
4. Data Preparation
   - Pemrosesan Data :
       - One-Hot Encoding untuk variabel kategorikal seperti JobRole, Department, dan Attrition.
       - Pembagian data menjadi data pelatihan dan pengujian dengan rasio 70:30.
       - Normalisasi data numerik menggunakan StandardScaler untuk memudahkan pelatihan model dan meningkatkan kinerja model deep learning.
  - Alasan : Tahapan data preparation penting untuk memastikan bahwa model dapat belajar dengan efektif. One-Hot Encoding memastikan bahwa data kategorikal dapat diproses oleh model machine learning,
    sementara normalisasi diperlukan untuk menghindari bias akibat perbedaan skala antar fitur.

5. Modelling
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
    
6. Evaluation
   - Metriks :
       - Accuracy: Persentase prediksi yang benar dari total prediksi.
       - Precision: Proporsi prediksi positif yang benar.
       - Recall: Proporsi karyawan yang benar-benar meninggalkan perusahaan dan diprediksi dengan benar.
       - F1-Score: Rata-rata harmonis antara precision dan recall.
   - Hasil Evaluasi : ```Model deep learning (ANN) memberikan hasil evaluasi yang baik dengan accuracy sekitar 82% pada data validasi. F1-Score digunakan untuk menilai keseimbangan antara precision dan recall.```
