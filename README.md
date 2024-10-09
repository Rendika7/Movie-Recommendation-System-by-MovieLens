# **Laporan Proyek Machine Learning - Rendika Nurhartanto Suharto**

## **Project Overview**

Proyek ini berfokus pada pengembangan **Movie Recommendation System** menggunakan dataset **MovieLens**, yang berisi metadata lengkap dari 45,000 film yang dirilis hingga Juli 2017. Dataset ini tidak hanya mencakup informasi tentang film seperti cast, crew, plot keywords, budget, dan pendapatan, tetapi juga menyertakan 26 juta rating dari 270,000 pengguna. Rating tersebut berada pada skala 1-5, dan diambil dari website resmi GroupLens.

Sistem rekomendasi memiliki peran penting dalam industri modern, terutama dalam meningkatkan pengalaman pengguna dengan memberikan rekomendasi yang lebih personal. Dengan volume data yang sangat besar seperti dalam dataset MovieLens, pengembangan algoritma rekomendasi yang efisien dapat memberikan keuntungan signifikan bagi platform seperti Netflix, Amazon Prime, atau aplikasi streaming lainnya.

Proyek ini penting karena dapat membantu mengatasi masalah information overload. Di era digital, pengguna sering kali merasa kewalahan dengan banyaknya pilihan konten yang tersedia. Dengan menggunakan sistem rekomendasi berbasis konten dan kolaboratif, pengguna dapat dengan mudah menemukan film yang sesuai dengan preferensi mereka berdasarkan sejarah rating dan metadata film yang ada.

### Hasil Riset Terkait
Beberapa penelitian terkait menunjukkan bahwa sistem rekomendasi berbasis machine learning mampu meningkatkan engagement pengguna secara signifikan. Studi seperti “**Matrix Factorization Technique for Recommendation Systems**” ([Koren et al., 2009](https://datajobs.com/data-science-repo/Recommender-Systems-%5BNetflix%5D.pdf)) menunjukkan bahwa pendekatan ini tidak hanya meningkatkan akurasi rekomendasi tetapi juga dapat mengurangi computational cost pada platform dengan skala besar.

Studi lain oleh Badrul Sarwar et al. (2001) berjudul “**Item-Based Collaborative Filtering Recommendation Algorithms**” menyajikan pendekatan berbasis item yang memungkinkan sistem untuk memberikan rekomendasi dengan memanfaatkan kesamaan antara item yang telah dinilai oleh pengguna. Metode ini menunjukkan efektivitas yang lebih baik dibandingkan pendekatan berbasis pengguna, khususnya dalam konteks data sparsity, dan dapat meningkatkan kepuasan pengguna dengan memberikan rekomendasi yang lebih relevan ([Sarwar et al., 2001](https://www.researchgate.net/publication/2369002_Item-based_Collaborative_Filtering_Recommendation_Algorithms)).

Selain itu, penelitian oleh Yifan Hu et al. (2008) dalam artikel berjudul “**Collaborative Filtering for Implicit Feedback Datasets**” membahas teknik baru dalam collaborative filtering yang dirancang untuk dataset dengan umpan balik tidak langsung, seperti yang terdapat dalam banyak sistem rekomendasi film. Pendekatan ini menggunakan model probabilistik untuk mengatasi tantangan yang dihadapi oleh sistem tradisional, dan telah terbukti efektif dalam meningkatkan kualitas rekomendasi serta memanfaatkan data pengguna secara lebih optimal ([Hu et al., 2008](https://ieeexplore.ieee.org/document/4781121)). 

Dengan memanfaatkan hasil penelitian ini, proyek ini bertujuan untuk mengembangkan sistem rekomendasi yang lebih efektif dan efisien, meningkatkan pengalaman pengguna dalam menjelajahi konten film.

### Referensi
- Koren et al., (2009) [Matrix Factorization Technique for Recommendation Systems](https://datajobs.com/data-science-repo/Recommender-Systems-%5BNetflix%5D.pdf) 
- Badrul Sarwar et al. (2001) [Item-Based Collaborative Filtering Recommendation Algorithms](https://www.researchgate.net/publication/2369002_Item-based_Collaborative_Filtering_Recommendation_Algorithms)
- Yifan Hu et al. (2008) [Collaborative Filtering for Implicit Feedback Datasets](https://ieeexplore.ieee.org/document/4781121) 


## **Business Understanding**

Pada bagian ini, proses klarifikasi masalah dalam pengembangan **Movie Recommendation System** dengan menggunakan dataset **MovieLens** akan dijelaskan secara rinci.

### Problem Statements

1. **Pernyataan Masalah 1**: Pengguna sering kali merasa kesulitan dalam menemukan film yang sesuai dengan preferensi mereka karena banyaknya pilihan yang tersedia.
   
2. **Pernyataan Masalah 2**: Rekomendasi film yang tidak relevan dapat mengurangi kepuasan pengguna dan membuat mereka kehilangan minat pada platform.

### Goals

- **Jawaban pernyataan masalah 1**: Membangun sistem rekomendasi yang dapat menganalisis preferensi pengguna berdasarkan rating sebelumnya dan metadata film untuk memberikan rekomendasi yang lebih relevan.
  
- **Jawaban pernyataan masalah 2**: Mengimplementasikan algoritma yang mempertimbangkan kesamaan antara film yang telah dinilai oleh pengguna lain untuk meningkatkan relevansi rekomendasi dan meminimalisir ketidakpuasan pengguna.

### Solution Approach

Untuk mencapai tujuan yang telah ditetapkan, beberapa pendekatan solusi yang dapat dipertimbangkan adalah:

1. **Sistem Rekomendasi Berbasis Konten**: Pendekatan ini menggunakan fitur-fitur dari film (seperti genre, sutradara, dan aktor) untuk memberikan rekomendasi. Algoritma seperti **TF-IDF** atau **Cosine Similarity** dapat digunakan untuk menghitung kesamaan antara film berdasarkan fitur-fitur ini, sehingga pengguna dapat menerima rekomendasi film yang mirip dengan yang telah mereka sukai sebelumnya.

2. **Sistem Rekomendasi Kolaboratif**: Dalam pendekatan ini, algoritma seperti **Matrix Factorization** (contohnya **SVD - Singular Value Decomposition**) dapat digunakan untuk menganalisis rating pengguna. Dengan cara ini, sistem dapat menemukan pola di antara pengguna dan film, sehingga rekomendasi dapat diberikan berdasarkan rating yang diberikan oleh pengguna dengan preferensi yang mirip.

Dengan merumuskan masalah dan tujuan secara jelas, serta mempertimbangkan berbagai pendekatan solusi, proyek ini bertujuan untuk memberikan pengalaman pengguna yang lebih baik dalam menemukan film yang sesuai dengan selera mereka.

## **Data Understanding**

Dataset yang digunakan dalam proyek ini adalah *Full MovieLens Dataset*, yang mencakup metadata untuk 45.000 film yang dirilis hingga Juli 2017. Dataset ini terdiri dari beberapa file, termasuk informasi tentang pemeran, kru, kata kunci plot, anggaran, pendapatan, poster, tanggal rilis, bahasa, perusahaan produksi, negara, jumlah suara TMDB, dan rata-rata suara. Selain itu, dataset ini juga mencakup 26 juta rating dari 270.000 pengguna untuk semua 45.000 film yang ada dalam dataset ini. Rating diberikan dalam skala 1-5 dan diperoleh dari situs resmi GroupLens. Anda dapat mengunduh dataset ini melalui tautan berikut: [MovieLens Dataset](https://grouplens.org/datasets/movielens/).

### Variabel-variabel pada Dataset

Dataset terdiri dari beberapa file yang menyimpan berbagai informasi tentang film. Berikut adalah penjelasan mengenai setiap file dan variabel di dalamnya:

| **Nama File**           | **Variabel**             | **Deskripsi**                                                                                       |
|-------------------------|--------------------------|-----------------------------------------------------------------------------------------------------|
| **movies_metadata.csv**  | adult                    | Menunjukkan apakah film ditujukan untuk dewasa (True/False).                                         |
|                         | belongs_to_collection     | Informasi tentang koleksi film, jika film merupakan bagian dari koleksi.                             |
|                         | budget                   | Anggaran produksi film.                                                                              |
|                         | genres                   | Genre film yang disajikan dalam bentuk JSON.                                                         |
|                         | homepage                 | URL homepage resmi film.                                                                             |
|                         | id                       | ID unik untuk film.                                                                                  |
|                         | imdb_id                  | ID film di IMDb.                                                                                     |
|                         | original_language        | Bahasa asli film.                                                                                    |
|                         | original_title           | Judul asli film.                                                                                     |
|                         | overview                 | Deskripsi singkat tentang film.                                                                      |
|                         | popularity               | Skor popularitas film.                                                                               |
|                         | poster_path              | Jalur ke gambar poster film.                                                                         |
|                         | production_companies     | Perusahaan produksi yang terlibat dalam pembuatan film.                                               |
|                         | production_countries     | Negara tempat film diproduksi.                                                                       |
|                         | release_date             | Tanggal rilis film.                                                                                  |
|                         | revenue                  | Pendapatan yang diperoleh dari film.                                                                 |
|                         | runtime                  | Durasi film dalam menit.                                                                             |
|                         | spoken_languages         | Bahasa yang digunakan dalam film.                                                                    |
|                         | status                   | Status rilis film (misalnya, "Released").                                                            |
|                         | tagline                  | Tagline film.                                                                                        |
|                         | title                    | Judul film.                                                                                          |
|                         | video                    | Menunjukkan apakah film memiliki video (True/False).                                                 |
|                         | vote_average             | Rata-rata rating film.                                                                               |
|                         | vote_count               | Jumlah total rating yang diterima film.                                                              |
| **keywords.csv**         | id                       | ID film yang berkaitan.                                                                              |
|                         | keywords                 | Kata kunci plot yang terkait dengan film dalam bentuk JSON.                                           |
| **credits.csv**          | cast                     | Informasi tentang pemeran film dalam bentuk JSON.                                                    |
|                         | crew                     | Informasi tentang kru film dalam bentuk JSON.                                                        |
|                         | id                       | ID film yang berkaitan.                                                                              |
| **links.csv**            | movieId                  | ID unik film.                                                                                        |
|                         | imdbId                   | ID film di IMDb.                                                                                     |
|                         | tmdbId                   | ID film di TMDb.                                                                                     |
| **links_small.csv**      | movieId                  | ID unik film untuk subset kecil dari 9.000 film.                                                     |
|                         | imdbId                   | ID film di IMDb.                                                                                     |
|                         | tmdbId                   | ID film di TMDb untuk subset kecil dari 9.000 film.                                                  |
| **ratings_small.csv**    | userId                   | ID unik pengguna yang memberikan rating.                                                             |
|                         | movieId                  | ID unik film yang dinilai.                                                                           |
|                         | rating                   | Nilai rating yang diberikan oleh pengguna (1-5).                                                     |
|                         | timestamp                | Waktu saat rating diberikan.                                                                         |

### Tahapan Pemahaman Dataset

Untuk lebih memahami dataset, kita dapat melakukan beberapa tahap analisis eksploratori (EDA) dan visualisasi data. Contoh analisis yang telah dilakukan meliputi:

### 1. movies_metadata.csv

Pada dataset `movies_metadata.csv`, dilakukan eksplorasi data awal untuk mengetahui ukuran dataset dan informasi dasar terkait missing values serta keunikan nilai di setiap kolom.

- **Ukuran Dataset:**
  Dataset ini memiliki ukuran (45466 Baris, 24 Kolom) yang terdiri dari berbagai fitur terkait metadata film, yang dapat dilihat dari output `md.shape`.

- **Visualisasi Missing Values:**
  Dilakukan visualisasi menggunakan heatmap untuk mengidentifikasi missing values di dataset. Gambar di bawah menunjukkan pola missing values, yang membantu dalam mengidentifikasi kolom-kolom yang mungkin perlu diperhatikan lebih lanjut untuk imputation atau penghapusan.

    ![Missing Values](https://github.com/Rendika7/Movie-Recommendation-System-by-MovieLens/blob/main/source/Missing%20Data.png?raw=true)

  Insight: Missing data tersebar di beberapa kolom, dan pola ini memberikan indikasi tentang kualitas data serta kebutuhan akan teknik penanganan missing data.

- **Unik (Uniqueness) pada Setiap Kolom:**
  Untuk setiap kolom numerik, dihitung nilai unik (uniqueness) serta persentase keunikan dibandingkan jumlah total baris. Kolom yang memiliki nilai unik <= 6 difilter dan divisualisasikan menggunakan bar plot. Kolom yang divisualkan adalah adult, status, dan video.

    ![Status Distribution](https://github.com/Rendika7/Movie-Recommendation-System-by-MovieLens/blob/main/source/Status-visual.png?raw=true)
    ![Video Distribution](https://github.com/Rendika7/Movie-Recommendation-System-by-MovieLens/blob/main/source/Video-status.png?raw=true)
    
  Insight: Beberapa kolom memiliki variasi yang sangat rendah, yang berarti kemungkinan data tersebut tidak memberikan banyak informasi, seperti kolom dengan nilai tetap atau kategori yang sangat terbatas.

### 2. links_small.csv

- **Ukuran Dataset:**
  Dataset ini lebih kecil dibandingkan dengan `movies_metadata.csv` yaitu sebesar (9125 Baris, 3 Kolom) dan digunakan sebagai penghubung antara data film dengan database lain. Informasi dasar dari dataset ditampilkan menggunakan fungsi `basic_data_info`.

    ![Distribusi Data - Link Small](https://github.com/Rendika7/Movie-Recommendation-System-by-MovieLens/blob/main/source/Tipe%20Data%20-%20Link%20Small.png?raw=true)
    ![Statistika Deskriptif](https://github.com/Rendika7/Movie-Recommendation-System-by-MovieLens/blob/main/source/Statistika%20Deskriptif%20-%20Link%20Small.png?raw=true)

  Insight: Dataset ini berfungsi sebagai kunci hubungan dengan dataset lainnya melalui ID film.

### 3. credits.csv

- **Ukuran Dataset:**
  Dataset ini berisi informasi terkait cast dan crew setiap film, dengan berbagai kolom yang menggambarkan aktor, sutradara, dan lainnya. Dataset sebesar 45476 baris, dan 3 Kolom.

- **Distribusi Tipe Data**
  Menampilkan informasi ringkas seperti tipe data dari tiap kolom.

    ![Tipe Data - credits](https://github.com/Rendika7/Movie-Recommendation-System-by-MovieLens/blob/main/source/Tipe%20Data%20-%20Credits.png?raw=true)

### 4. keywords.csv

- **Ukuran Dataset:**
  Dataset ini berisi daftar kata kunci yang terkait dengan setiap film, yang dapat digunakan untuk analisis lebih lanjut dalam sistem rekomendasi berbasis konten. Dataset ini berukuran sebesar 46419 Baris, dan 2 Kolom.

- **Word Cloud:**
  Untuk visualisasi, dibangun word cloud yang menunjukkan kata kunci paling umum yang muncul dalam dataset.

    ![Tipe Data - credits](https://github.com/Rendika7/Movie-Recommendation-System-by-MovieLens/blob/main/source/Wordcloud.png?raw=true)

  Insight: Kata kunci yang muncul paling sering menggambarkan genre atau tema yang sering muncul dalam film, dan ini penting untuk membangun rekomendasi berdasarkan konten film.

### 5. ratings_small.csv

- **Ukuran Dataset:**
  Dataset ini berisi informasi terkait rating yang diberikan oleh pengguna terhadap film-film tertentu. Dataset ini sangat penting untuk analisis sistem rekomendasi berbasis collaborative filtering. Dataset disini berukuran 100004 Baris, dan 4 Kolom.

- **Distribusi Rating:**
  Untuk memahami distribusi rating, dibuat dua jenis plot, yaitu bar plot dan KDE plot, yang membantu memahami frekuensi dan kepadatan distribusi rating.

    ![Distribusi Rating](https://github.com/Rendika7/Movie-Recommendation-System-by-MovieLens/blob/main/source/Distribution%20Rating.png?raw=true)

  Insight: Rating cenderung berkumpul di sekitar nilai-nilai tertentu, memberikan gambaran bahwa sebagian besar film mendapatkan rating rata-rata hingga tinggi.

Bab ini mencakup pemahaman awal terkait dataset yang digunakan dengan teknik-teknik visualisasi data untuk membantu memahami pola dan distribusi dalam dataset, serta insight yang dapat berguna untuk analisis lebih lanjut.

## **Data Preparation**

Data preparation merupakan langkah penting dalam proses pembangunan sistem rekomendasi, karena memastikan bahwa data yang digunakan bersih, konsisten, dan siap untuk diolah lebih lanjut. Pada proyek ini, kami melakukan beberapa teknik data preparation, antara lain:

### 1. Data Cleaning
Data cleaning bertujuan untuk menghilangkan data yang tidak konsisten atau tidak diperlukan, serta menangani nilai-nilai kosong. Langkah-langkah yang dilakukan dalam data cleaning antara lain:

- **Menghapus Baris dengan Nilai Null pada Kolom 'tmdbId'**  
  Kode berikut digunakan untuk menghapus nilai null pada kolom `tmdbId` di DataFrame `links_small`, serta mengonversi kolom tersebut menjadi tipe integer.
  ```python
  links_small = links_small[links_small['tmdbId'].notnull()]['tmdbId'].astype('int')
  ```

- **Menghapus Baris yang Tidak Sesuai pada Kolom 'adult'**  
  Dalam DataFrame `md`, baris-baris yang memiliki nilai tidak sesuai dalam kolom `adult` dihapus.
  ```python
  md = md.drop([19730, 29503, 35587])
  ```

- **Mengubah Tipe Data Kolom 'id'**  
  Kolom `id` diubah menjadi tipe data integer untuk memastikan konsistensi dalam pengolahan data selanjutnya.
  ```python
  md['id'] = md['id'].astype('int')
  ```

### 2. Data Selection
Pada langkah ini, kami melakukan seleksi data untuk mendapatkan subset data yang relevan. Beberapa proses yang dilakukan:

- **Menyaring Data berdasarkan Kesamaan ID**  
  Kode berikut digunakan untuk menghasilkan DataFrame baru (`smd`) yang hanya berisi baris dari `md` di mana `id` sesuai dengan nilai yang terdapat pada kolom `tmdbId` dari DataFrame `links_small`.
  ```python
  smd = md[md['id'].isin(links_small)]
  ```

### 3. Handling Missing Values
Pada proses ini, kami menangani nilai kosong (missing values) dalam beberapa kolom yang penting. Beberapa langkah yang dilakukan:

- **Mengisi Nilai Kosong pada Kolom 'tagline' dengan String Kosong**  
  Untuk memastikan tidak ada nilai kosong pada kolom `tagline`, nilai kosong diganti dengan string kosong (`''`).
  ```python
  smd['tagline'] = smd['tagline'].fillna('')
  ```

- **Menggabungkan Kolom 'overview' dan 'tagline' menjadi 'description'**  
  Setelah mengisi nilai kosong, kolom `overview` dan `tagline` digabungkan untuk membentuk kolom `description`, sehingga memberikan deskripsi yang lebih lengkap untuk setiap film.
  ```python
  smd['description'] = smd['overview'] + smd['tagline']
  smd['description'] = smd['description'].fillna('')
  ```

### 4. Feature Engineering
Proses ini melibatkan pembuatan fitur-fitur baru yang dapat memperkaya data dan meningkatkan performa sistem rekomendasi. Beberapa teknik feature engineering yang dilakukan adalah:

- **Membuat Kolom 'combined_clean'**  
  Kami menggabungkan beberapa kolom penting seperti `genres`, `cast`, `director`, dan `keywords` untuk membuat satu kolom gabungan bernama `combined_clean`. Kolom ini akan menjadi representasi komprehensif dari metadata setiap film.
  ```python
  smd['combined_clean'] = smd['tagline'] + ' | ' + smd['description']
  ```

- **Merapikan Nama dan Menormalkan Data**  
  Kami melakukan transformasi pada data untuk menghindari ketidakkonsistenan seperti perbedaan penggunaan huruf kapital dan spasi. Misalnya, kode berikut mengubah semua nama aktor dan sutradara menjadi huruf kecil dan menghapus spasi di antaranya:
  ```python
  smd['cast'] = smd['cast'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])
  smd['director'] = smd['director'].astype('str').apply(lambda x: str.lower(x.replace(" ", "")))
  ```

- **Stemming pada Kata Kunci**  
  Kami juga melakukan stemming pada kolom `keywords` untuk memastikan kata-kata yang berbeda tapi memiliki makna dasar yang sama dihitung sebagai satu entitas. Teknik ini membantu dalam menjaga konsistensi data.
  ```python
  smd['keywords'] = smd['keywords'].apply(lambda x: [stemmer.stem(word) for word in x])
  ```

### 5. Data Transformation
Pada tahap ini, kami menggunakan berbagai teknik transformasi data untuk mengubah teks menjadi format yang dapat dianalisis oleh algoritma rekomendasi. Langkah-langkah yang diambil:

- **TF-IDF Transformation pada Kolom Deskripsi Film**  
  Kami menggunakan **TfidfVectorizer** untuk mengubah kolom `description` menjadi matriks TF-IDF. Matriks ini mewakili pentingnya setiap kata dalam deskripsi film.
  ```python
  tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2), min_df=0.0, stop_words='english')
  tfidf_matrix = tf.fit_transform(smd['description'])
  ```

- **Count Vectorization pada Metadata Film**  
  Selain menggunakan TF-IDF, kami juga melakukan **Count Vectorization** untuk mengubah metadata seperti `cast`, `director`, `genres`, dan `keywords` menjadi fitur numerik yang bisa digunakan dalam perhitungan kesamaan film.
  ```python
  count = CountVectorizer(analyzer='word',ngram_range=(1, 2), min_df=0.0, stop_words='english')
  count_matrix = count.fit_transform(smd['combined_clean'])
  ```

### 6. Final Dataset Preparation
Pada tahap akhir, kami menggabungkan semua transformasi yang telah dilakukan untuk membuat dataset akhir yang siap digunakan dalam model sistem rekomendasi. Proses ini mencakup pembuatan indeks untuk setiap film, yang akan digunakan untuk mengembalikan rekomendasi berdasarkan film yang dipilih pengguna.
```python
smd = smd.reset_index()
titles = smd[['title', 'combined_clean']]
indices = pd.Series(smd.index, index=smd['title'])
```
Teknik data preparation ini memastikan bahwa dataset yang digunakan sudah siap untuk dianalisis dan diolah oleh model.

## **MODELING**

Pada tahap ini, kami akan membahas dua jenis sistem rekomendasi yang digunakan dalam proyek ini, yaitu **Content-Based Recommender** dan **Collaborative Filtering**. Tujuan dari tahap ini adalah untuk menghasilkan Top-N recommendation yang memanfaatkan informasi metadata film dan perilaku pengguna. Berikut adalah penjelasan lebih mendetail mengenai masing-masing pendekatan:

### 1. Content-Based Recommender

Sistem rekomendasi berbasis konten bertujuan untuk merekomendasikan film kepada pengguna berdasarkan karakteristik deskriptif dari film itu sendiri, seperti genre, sutradara, aktor, serta sinopsis. Dalam pendekatan ini, ada dua jenis rekomendasi berbasis konten yang diimplementasikan:

#### 1.1 Movie Description Based Recommender

Langkah pertama dalam pembuatan sistem rekomendasi berbasis deskripsi film adalah melakukan preprocessing pada data deskripsi film. Beberapa tahapan yang dilakukan meliputi:

1. Penggabungan Overview dan Tagline:
   Untuk setiap film, kolom `overview` (deskripsi film) dan `tagline` (slogan film) digabungkan menjadi satu kolom baru yang disebut `description`. Ini dilakukan untuk membuat deskripsi yang lebih lengkap dari setiap film.

2. TF-IDF Vectorizer:
   Kami menggunakan teknik **TF-IDF (Term Frequency-Inverse Document Frequency)** untuk mengubah teks dalam deskripsi film menjadi representasi numerik yang dapat digunakan untuk perhitungan kesamaan. Dengan pengaturan seperti:
   - `analyzer='word'`: Menganalisis teks pada tingkat kata.
   - `ngram_range=(1, 2)`: Membentuk unigram dan bigram.
   - `stop_words='english'`: Mengabaikan kata-kata umum dalam bahasa Inggris.

   Hasilnya adalah matriks **TF-IDF** yang kemudian digunakan untuk menghitung kesamaan antar film menggunakan **cosine similarity**.

Tentu, saya akan membetulkan format rumus agar bisa tampil dengan benar. Berikut adalah versi yang sudah diperbaiki:

---

3. Cosine Similarity:
Cosine similarity digunakan untuk mengukur kemiripan antara setiap deskripsi film dalam dataset. Matriks kesamaan yang dihasilkan menunjukkan seberapa mirip satu film dengan yang lainnya, dengan nilai antara 0 (tidak mirip) hingga 1 (sangat mirip).

- Rumus Cosine Similarity:
Secara matematis, cosine similarity antara dua vektor \( A \) dan \( B \) didefinisikan sebagai:

  $$
  \text{cosine similarity}(A, B) = \frac{A \cdot B}{\|A\| \|B\|}
  $$
  
  - \( A dot B \) adalah **dot product** dari vektor \( A \) dan \( B \).
  - \( \|A\| \) dan \( \|B\| \) adalah **norma (magnitudo)** dari vektor \( A \) dan \( B \), masing-masing, dihitung dengan rumus:
  
  $$
  \|A\| = \sqrt{\sum_{i=1}^{n} A_i^2} \quad \text{dan} \quad \|B\| = \sqrt{\sum_{i=1}^{n} B_i^2}
  $$
  
  Jadi, rumus lengkapnya menjadi:
  
  $$
  \text{cosine similarity}(A, B) = \frac{\sum_{i=1}^{n} A_i B_i}{\sqrt{\sum_{i=1}^{n} A_i^2} \times \sqrt{\sum_{i=1}^{n} B_i^2}}
  $$

- Kelebihan Cosine Similarity:
  - **Skalabilitas**: Cocok untuk data berukuran besar karena tidak terlalu bergantung pada magnitudo data.
  - **Efisiensi**: Hanya memerlukan operasi dot product dan norma, yang membuatnya efisien dihitung dalam ruang berdimensi tinggi seperti dalam dokumen teks atau metadata film.

Dengan cosine similarity, kita bisa menghitung dan merekomendasikan film-film yang memiliki karakteristik atau deskripsi yang paling mirip dengan preferensi pengguna atau film lain.

4. Fungsi Rekomendasi:
   Kami menulis fungsi `get_recommendations()` yang mengambil judul film dan mengembalikan 10 film paling mirip berdasarkan skor kesamaan kosinus. Sebagai contoh, jika input judul film adalah *The Dark Knight*, maka sistem akan mengembalikan 10 film lain yang memiliki deskripsi dan metadata yang mirip.

    ![Function Get Recom](https://github.com/Rendika7/Movie-Recommendation-System-by-MovieLens/blob/main/source/get_recommendations()%20Function.png?raw=true)

    ![Hasil Rekomendasi](https://github.com/Rendika7/Movie-Recommendation-System-by-MovieLens/blob/main/source/Hasil%20Rekomendasi%201.png?raw=true)

#### 1.2 Metadata Based Recommender

Selain rekomendasi berbasis deskripsi film, kami juga menggunakan metadata lain seperti:
- **Aktor**, 
- **Sutradara**, 
- **Kata kunci (keywords)**, 
- **Genre**.

Tahapan yang dilakukan untuk mempersiapkan data metadata adalah sebagai berikut:

1. Pengolahan Data Metadata:
   Kolom-kolom seperti `cast`, `crew`, dan `keywords` diproses menggunakan fungsi **literal_eval** untuk mengubah nilai string menjadi list, sehingga setiap elemen dapat diakses secara terpisah.

2. Pembersihan Data:
   Setiap fitur diubah menjadi huruf kecil dan dihapus spasinya. Ini bertujuan untuk menghindari kesalahan ketika ada aktor atau sutradara yang memiliki nama serupa. Misalnya, nama *Johnny Depp* akan dianggap sama dengan *Johnny depp*.

3. Pemberian Bobot pada Sutradara:
   Sutradara disebutkan sebanyak tiga kali dalam kombinasi metadata untuk memberikan bobot lebih pada pengaruh sutradara dalam film.

4. Matriks Count Vectorizer:
   Dengan menggunakan **CountVectorizer**, kami membentuk matriks dari metadata gabungan untuk setiap film. Setelah itu, cosine similarity kembali digunakan untuk menghitung kesamaan antara film-film berdasarkan metadata ini.

5. Fungsi Rekomendasi dan Hasil:
   Fungsi yang digunakan sama seperti pada metode 4.1.1 yang dimana akan mengeluarkan hasil seperti dibawah ini.

    ![Hasil Rekomendasi](https://github.com/Rendika7/Movie-Recommendation-System-by-MovieLens/blob/main/source/Hasil%20Rekomendasi%202.png?raw=true)

### 2. Collaborative Filtering

Selain pendekatan berbasis konten, kami juga menggunakan metode **Collaborative Filtering**. Metode ini memprediksi preferensi pengguna berdasarkan pola rating dari pengguna lain yang mirip.

#### 2.1 Implementasi SVD

1. **Data Loading**:
   Data rating dimuat menggunakan pustaka **Surprise** dan kami menggunakan algoritma **SVD (Singular Value Decomposition)** untuk membangun model rekomendasi berdasarkan data rating dari pengguna.

2. **Evaluasi Model**:
   Kami melakukan cross-validation menggunakan metrik **RMSE** (Root Mean Squared Error) dan **MAE** (Mean Absolute Error). Hasil rata-rata dari 10 kali cross-validation menunjukkan bahwa model memiliki kinerja yang baik dengan nilai RMSE dan MAE yang rendah.

3. **Prediksi Rating**:
   Setelah model dilatih, kami melakukan prediksi rating untuk film-film yang belum ditonton oleh pengguna tertentu (misalnya pengguna dengan userId = 1). Model akan memprediksi seberapa besar kemungkinan pengguna menyukai film yang belum ditonton berdasarkan pola rating pengguna lain yang mirip.

4. **Top-N Recommendations**:
   Berdasarkan hasil prediksi, kami mengurutkan film-film yang belum ditonton dan memilih 10 film dengan prediksi rating tertinggi sebagai rekomendasi untuk pengguna.

---

### Kesimpulan Modeling

Pada tahap ini, kami berhasil membangun dua sistem rekomendasi yang memanfaatkan informasi deskriptif film (content-based) dan perilaku pengguna (collaborative filtering). Hasil rekomendasi telah diuji dan menunjukkan tingkat presisi yang baik, terutama untuk film dengan deskripsi dan metadata yang kaya.

Sistem rekomendasi berbasis konten mampu memberikan rekomendasi yang relevan berdasarkan karakteristik film itu sendiri, sementara collaborative filtering memberikan rekomendasi yang lebih personal berdasarkan pola rating pengguna lain. Keduanya digabungkan untuk menghasilkan sistem rekomendasi yang lebih holistik dan akurat.




## Evaluation
Pada bagian ini Anda perlu menyebutkan metrik evaluasi yang digunakan. Kemudian, jelaskan hasil proyek berdasarkan metrik evaluasi tersebut.

Ingatlah, metrik evaluasi yang digunakan harus sesuai dengan konteks data, problem statement, dan solusi yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja.
