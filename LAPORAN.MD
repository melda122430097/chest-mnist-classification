# 1. PENDAHULUAN

Perkembangan teknologi kecerdasan buatan (Artificial Intelligence/AI) telah memberikan dampak signifikan dalam bidang kesehatan, khususnya pada analisis citra medis. Salah satu metode yang paling berkembang adalah Deep Learning, yang memungkinkan komputer mempelajari pola-pola kompleks pada gambar melalui jaringan saraf dalam (deep neural networks). Teknologi ini sangat membantu dalam proses diagnostik, terutama pada penyakit-penyakit kritis yang memerlukan identifikasi cepat dan akurat. Salah satu gambaran medis yang paling sering digunakan dalam penegakan diagnosis adalah citra X-ray dada (Chest X-Ray). Citra ini memuat informasi mengenai organ thoraks seperti jantung, paru-paru, dan rongga pleura. Kelainan tertentu seperti Cardiomegaly (pembesaran jantung) dan Pneumothorax (kolaps paru akibat udara di rongga pleura) dapat dideteksi melalui pola visual radiografis pada gambar tersebut.

Untuk membantu proses deteksi otomatis kedua penyakit tersebut, penelitian ini menggunakan dataset ChestMNIST, yang merupakan bagian dari kumpulan dataset MedMNIST. Dataset ini memiliki karakteristik multi-label, sehingga satu citra dapat menunjukkan lebih dari satu penyakit. Agar model Deep Learning dapat belajar dengan optimal, dataset perlu diseleksi, disaring, dan diproses menjadi dataset klasifikasi biner untuk dua penyakit tersebut. Penelitian ini tidak hanya membahas pengolahan data, tetapi juga membandingkan tiga arsitektur model, yaitu SimpleCNN, ResNet, dan DenseNet, untuk mengetahui model yang memberikan kinerja terbaik dalam mengenali kedua penyakit tersebut.

# 2. TINJAUAN TEORI

2.1 Citra X-Ray Dada

Citra X-ray dada merupakan teknik pencitraan medis yang memanfaatkan radiasi sinar-X untuk menggambarkan struktur organ dalam bagian thoraks. Proses ini menghasilkan bayangan dengan tingkat kecerahan berbeda sesuai dengan densitas jaringan yang dilewati sinar-X. Jaringan yang padat seperti tulang akan tampak lebih terang, sedangkan jaringan yang lebih lunak seperti paru-paru akan terlihat lebih gelap. Melalui perbedaan intensitas inilah dokter dapat mengidentifikasi kondisi abnormal, seperti penyempitan, penumpukan cairan, atau kerusakan jaringan. Oleh karena itu, citra X-ray dada menjadi salah satu alat diagnostik utama dalam evaluasi penyakit paru dan jantung.

2.2 Cardiomegaly

Cardiomegaly adalah kondisi medis di mana ukuran jantung tampak membesar ketika dilihat melalui citra X-ray dada. Pembesaran ini biasanya terlihat dari melebar atau membesarnya siluet jantung melebihi proporsi normal terhadap rongga dada. Cardiomegaly dapat menjadi indikator adanya gangguan pada otot jantung, tekanan darah tinggi kronis, atau kondisi gagal jantung. Keberadaan pembesaran jantung ini penting untuk diperhatikan, karena dapat mempengaruhi kemampuan jantung dalam memompa darah secara optimal ke seluruh tubuh. Dengan demikian, deteksi dini Cardiomegaly melalui X-ray sangat penting untuk menentukan penanganan yang tepat.

2.3 Pneumothorax

Pneumothorax merupakan kondisi di mana terdapat udara yang masuk ke dalam rongga pleura, yaitu ruang antara paru-paru dan dinding dada. Masuknya udara ini menyebabkan paru-paru kehilangan tekanan yang diperlukan untuk mengembang, sehingga paru tampak kolaps sebagian atau bahkan sepenuhnya pada citra X-ray. Pada gambar X-ray, Pneumothorax biasanya ditandai dengan area yang tampak lebih gelap dan kosong karena tidak adanya jaringan atau aliran udara normal. Kondisi ini dapat terjadi akibat trauma fisik, prosedur medis, maupun secara spontan tanpa penyebab jelas. Pneumothorax merupakan kondisi darurat yang membutuhkan penanganan cepat karena dapat mengganggu fungsi pernapasan.

2.4 Deep Learning untuk Citra Medis

Deep Learning adalah pendekatan dalam kecerdasan buatan yang memanfaatkan jaringan saraf dalam untuk mempelajari pola kompleks dari data dalam jumlah besar. Dalam konteks citra medis, metode yang paling sering digunakan adalah Convolutional Neural Network (CNN), karena mampu mengenali pola visual seperti tekstur, bentuk, dan perbedaan intensitas pada gambar. CNN bekerja dengan mengekstraksi fitur citra melalui lapisan konvolusi secara bertahap dari fitur sederhana hingga fitur abstrak yang semakin kompleks. Hal ini membuat CNN sangat efektif dalam mendeteksi kelainan pada citra X-ray yang mungkin tidak terlihat jelas oleh pengamatan biasa. Oleh sebab itu, Deep Learning memiliki potensi besar dalam membantu tenaga medis melakukan diagnosis secara lebih cepat, objektif, dan akurat.

# 3. PENGOLAHAN DATA

3.1 Sumber Data

Dataset yang digunakan adalah ChestMNIST, berisi citra X-ray dalam format grayscale 28×28 piksel yang telah dianotasi dengan 14 label penyakit.

3.2 Penyederhanaan Kelas (Binary Classification)

Penelitian memfokuskan klasifikasi pada dua penyakit:

Label	Kondisi	Keterangan
1) Cardiomegaly	Pembesaran jantung
2) Pneumothorax	Paru kolaps akibat udara di rongga pleura

3.3 Penyaringan Single-Label

Karena dataset bersifat multi-label, maka hanya dipilih citra yang memiliki satu penyakit saja. Hal ini dilakukan untuk:

1) Menghindari ambiguitas pola visual
2) Memastikan label benar-benar mencerminkan kondisi pada citra
3) Meningkatkan akurasi proses pembelajaran model

3.4 Transformasi dan Normalisasi

Citra dinormalisasi untuk menstabilkan distribusi pixel sehingga mempercepat konvergensi model.

3.5 Pembagian Dataset

Dataset dibagi menjadi:

1) Training Set untuk pelatihan model
2) Validation Test untuk menguji generalisasi model


# 4. PELATIHAN MODEL DAN ANALISIS PERFORMA**

4.1 Arsitektur yang Dibandingkan
1) SimpleCNN
- Struktur jaringan sederhana
- Pelatihan cepat
- Kelemahan: Kapasitas representasi rendah → cenderung underfitting

2) ResNet
- Menggunakan skip-connection
- Mencegah vanishing gradient
- Kelebihan: Stabil pada jaringan dalam
- Namun: Performa masih fluktuatif tergantung pengaturan parameter

3) DenseNet
- Setiap layer terhubung ke seluruh layer sebelumnya
- Fitur yang sudah dipelajari tidak hilang
- Aliran gradien sangat stabil
- Paling efektif untuk pola radiologis halus

4.2 Hasil Akurasi 
Model	Kinerja Validasi	Penjelasan
- SimpleCNN	~64%	Kurang mampu menangkap detail visual
- ResNet	~71–75%	Lebih baik, namun masih fluktuatif
- DenseNet	~75–85% (Tertinggi)	Representasi fitur paling kuat dan stabil

4.3 Mengapa DenseNet Tertinggi?
DenseNet unggul karena:
- Setiap layer menerima fitur dari semua layer sebelumnya
→ Informasi visual penting tetap terjaga.
- Gradien mengalir tanpa hambatan
→ Menghindari hilangnya sinyal pembelajaran (vanishing gradient).
- Efektif pada citra dengan pola halus, seperti:
- Tepi jantung pada Cardiomegaly
- Area paru kolaps pada Pneumothorax
- Lebih efisien parameter dibanding ResNet
→ Lebih dalam, tetapi tidak boros memori.
- Sehingga DenseNet mampu membedakan kedua penyakit dengan akurasi tertinggi.

# 5. KESIMPULAN

Dataset ChestMNIST telah berhasil diproses menjadi dataset klasifikasi biner yang bersih, terstruktur, dan sesuai dengan tujuan penelitian, yaitu membedakan kondisi Cardiomegaly dan Pneumothorax secara spesifik tanpa adanya ambiguitas label. Hasil perbandingan terhadap beberapa arsitektur model menunjukkan bahwa kompleksitas struktur jaringan saraf memiliki pengaruh langsung terhadap kualitas performa klasifikasi. Model SimpleCNN, yang memiliki arsitektur relatif sederhana, cenderung mengalami underfitting karena kapasitas representasinya terbatas sehingga kurang mampu menangkap pola visual halus pada citra X-ray dada. Sementara itu, ResNet menunjukkan peningkatan performa melalui mekanisme skip connection yang membuat proses pembelajaran lebih stabil dan akurat dibandingkan SimpleCNN. Namun, hasil terbaik diperoleh dari model DenseNet, yang mampu memanfaatkan informasi fitur secara lebih maksimal melalui konektivitas antar-layer yang padat dan terarah. Arsitektur ini memungkinkan aliran gradien yang lebih stabil serta pelestarian detail fitur pada setiap tahap pemrosesan, sehingga DenseNet mampu mencapai akurasi tertinggi dalam mendeteksi Cardiomegaly dan Pneumothorax dibandingkan model-model lainnya.