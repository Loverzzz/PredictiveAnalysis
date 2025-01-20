# Laporan Proyek Machine Learning - Reynaldo Arya Budi Trisna


## Domain Proyek
  **Latar Belakang**:
  Masalah prediksi attrition karyawan (karyawan yang meninggalkan perusahaan) menjadi salah satu tantangan utama dalam manajemen SDM.
  Dengan memahami faktor-faktor yang mempengaruhi keputusan karyawan untuk meninggalkan perusahaan, perusahaan dapat mengambil langkah-langkah preventif untuk meningkatkan retensi karyawan.
  Dataset ini berisi informasi tentang karyawan dan status attrition mereka, yang bisa membantu perusahaan dalam merancang kebijakan yang lebih baik terkait dengan kepuasan karyawan dan pengelolaan tenaga kerja.
  
  Masalah ini penting karena tingginya tingkat pergantian karyawan dapat menambah biaya operasional perusahaan, seperti biaya rekrutmen dan pelatihan.
  Oleh karena itu, memprediksi attrition dengan akurat memungkinkan perusahaan untuk melakukan tindakan yang lebih efektif, seperti program pelatihan yang lebih baik, peningkatan kesejahteraan, dan pengelolaan beban kerja.

  **Referensi data**: https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset

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
      - Referensi data : https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset
  - Informasi Dataset
    Dataset ini berisi informasi mengenai 1.470 karyawan dengan 35 kolom yang mencakup faktor-faktor yang dapat mempengaruhi attrition, seperti:
    ```
    Age: Usia karyawan
    Attrition: Apakah karyawan meninggalkan perusahaan (1 = Yes, 0 = No)
    BusinessTravel: Frekuensi perjalanan dinas karyawan
    DailyRate: Tarif harian yang diterima karyawan
    Department: Departemen tempat karyawan bekerja
    DistanceFromHome: Jarak tempat tinggal karyawan dari kantor
    Education: Tingkat pendidikan karyawan
    EducationField: Bidang pendidikan karyawan
    EmployeeCount: Jumlah karyawan dalam perusahaan
    EmployeeNumber: Nomor identifikasi unik untuk karyawan
    EnvironmentSatisfaction: Kepuasan karyawan terhadap lingkungan kerja
    Gender: Jenis kelamin karyawan
    HourlyRate: Tarif per jam yang diterima karyawan
    JobInvolvement: Tingkat keterlibatan karyawan dalam pekerjaan
    JobLevel: Tingkat jabatan atau posisi karyawan
    JobRole: Peran pekerjaan karyawan
    JobSatisfaction: Kepuasan kerja karyawan
    MaritalStatus: Status pernikahan karyawan
    MonthlyIncome: Gaji bulanan karyawan
    MonthlyRate: Tarif bulanan yang diterima karyawan
    NumCompaniesWorked: Jumlah perusahaan tempat karyawan bekerja sebelumnya
    Over18: Menunjukkan apakah karyawan berusia di atas 18 tahun
    OverTime: Apakah karyawan bekerja lembur
    PercentSalaryHike: Persentase kenaikan gaji yang diterima karyawan
    PerformanceRating: Penilaian kinerja karyawan
    RelationshipSatisfaction: Kepuasan karyawan terhadap hubungan kerja di tempat kerja
    StandardHours: Jumlah jam kerja standar per minggu
    StockOptionLevel: Tingkat opsi saham yang diterima karyawan
    TotalWorkingYears: Total lama tahun kerja karyawan
    TrainingTimesLastYear: Jumlah pelatihan yang diikuti karyawan selama tahun terakhir
    WorkLifeBalance: Tingkat keseimbangan antara kehidupan pribadi dan pekerjaan karyawan
    YearsAtCompany: Jumlah tahun karyawan bekerja di perusahaan
    YearsInCurrentRole: Jumlah tahun karyawan bekerja di posisi atau peran saat ini
    YearsSinceLastPromotion: Jumlah tahun sejak karyawan terakhir dipromosikan
    YearsWithCurrManager: Jumlah tahun karyawan bekerja dengan manajer saat ini
    ```
  - Kondisi data : 0 missing Value dan 0 duplicated
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
    Hyperparameter:
     - Batch size : 32
     - Epoch : 200 with early stop Patience = 15
     - Optimizer Nadam
    
    
## Evaluation
   - Metriks :
       - Accuracy: Persentase prediksi yang benar dari total prediksi.
       - Precision: Proporsi prediksi positif yang benar.
       - Recall: Proporsi karyawan yang benar-benar meninggalkan perusahaan dan diprediksi dengan benar.
       - F1-Score: Rata-rata harmonis antara precision dan recall.
   - Hasil Evaluasi : ```Model deep learning (ANN) memberikan hasil evaluasi yang baik dengan accuracy sekitar 87% pada data validasi. Precision: 0.59 Recall: 0.41 F1-Score: 0.49.```
   - Menjawab Problem statement 1 : Model yang dikembangkan, yaitu Artificial Neural Network (ANN), mampu melakukan prediksi attrition karyawan dengan akurasi 87% pada data validasi. Model ini dapat digunakan untuk proses identifikasi cepat terhadap karyawan yang berisiko meninggalkan perusahaan, sehingga perusahaan dapat mengambil tindakan preventif lebih awal.
   - Menjawab Problem statement 2 : Model berhasil mengidentifikasi faktor-faktor utama yang memengaruhi attrition, seperti gaji bulanan (Monthly Income), usia (Age), pengalaman kerja (Total Working Years), dan faktor penting lainnya. Dengan mengetahui faktor-faktor ini, perusahaan dapat merancang kebijakan yang lebih efektif untuk menurunkan attrition karyawan.
   - Mencapai Goals 1 :
   - Membangun model prediksi attrition
     ```ANN yang digunakan memberikan hasil evaluasi yang baik (akurasi 87%, precision 0.59, recall 0.41). Ini menunjukkan bahwa model mampu memberikan wawasan untuk pengambilan keputusan berbasis data.```
   - Mencapai Goals 2 :
   - Mengidentifikasi faktor penting attrition 
     ```Analisis faktor dengan Random Forest Tree berhasil mengungkap variabel penting seperti gaji rendah, usia muda, dan pengalaman kerja singkat, yang menjadi penyebab utama attrition. Wawasan ini dapat membantu perusahaan merancang kebijakan strategis untuk meningkatkan retensi karyawan.```
   - Dampak Solusi :
   - Artificial Neural Network (ANN):
       - Dampak: Model ini mampu menangani hubungan non-linear antar variabel, memberikan prediksi yang akurat, dan membantu perusahaan mengidentifikasi karyawan yang berisiko tinggi untuk meninggalkan perusahaan. Akurasi 87% menunjukkan bahwa model ini cukup andal untuk implementasi.
       - Implementasi: Hasil prediksi dapat digunakan untuk identifikasi cepata dan melakukan preventif karyawan dari attrition.
   - Random Forest Tree:
       - Dampak :  Analisis faktor signifikan memberikan wawasan yang berharga bagi perusahaan untuk memahami akar penyebab attrition. Dengan informasi ini, perusahaan dapat mengambil langkah-langkah strategis, seperti meningkatkan kesejahteraan karyawan dengan gaji rendah atau perlakuan lainnya.
       - Implementasi: Wawasan dari analisis faktor ini dapat digunakan untuk mengetahui faktor faktor yang merupakan penyebab attrition dan merancang kebijakan jangka panjang ke depan.

**---Ini adalah bagian akhir laporan---**
