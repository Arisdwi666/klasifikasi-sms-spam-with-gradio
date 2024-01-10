Klasifikasi SMS Spam Berbahasa Indonesia Menggunakan Metode Multinomial Naive Bayes & Feature Selection Chi-Square dan Deploy menggunakan Gradio - Hosting menggunakan Hugging Face

Dataset diambil dari github https://github.com/ksnugroho/klasifikasi-spam-sms/ dengan jumlah data 1143 data. 569 data untuk SMS Normal, 335 data untuk Peniuan/Fraud, 239 data untuk Promo. 

Permasalahan : SMS spam adalah pesan yang tidak diinginkan atau tidak diminta oleh pengguna, yang dapat mengganggu, menipu, atau bahkan merugikan pengguna.
Tujuan : Untuk mengklasifikasikan spam SMS dan mengembangkan sebuah sistem klasifikasi SMS spam berbahasa Indonesia yang efektif dan akurat menggunakan metode Multinomial Naive Bayes.

Model : Menggunakan algoritma klasifikasi Multinomial Naive Bayes (MNB) untuk mengklasifikasikan SMS menjadi spam atau non-spam berdasarkan fitur-fitur yang diekstrak.
Langkah-langkah penyelesaian meliputi:
1. Data Acquisition
2. Text Pre-processing
   1. Case Folding
   2. Filtering
   3. Stopword
   4. Stemming
3. Feature Engineering
   1. Feature Extraction - BoW & TF IDF
   2. Feature Selection - Chi-Square
4. Modelling (Machine Learning)
5. Model Evaluation
6. Deployment

Performa Model : 
Jumlah prediksi benar	: 211
Jumlah prediksi salah	: 18
Akurasi pengujian	: 92.13973799126637 %
Confusion matrix:
 [[106   1   1]
 [  6  64   1]
 [  6   3  41]]

 Classification report:
               precision    recall  f1-score   support

           0       0.90      0.98      0.94       108
           1       0.94      0.90      0.92        71
           2       0.95      0.82      0.88        50

    accuracy                           0.92       229
   macro avg       0.93      0.90      0.91       229
weighted avg       0.92      0.92      0.92       229

Akurasi setiap split: [0.91266376 0.89956332 0.930131   0.89956332 0.91266376 0.91266376
 0.94759825 0.89519651 0.89519651 0.89082969] 

Rata-rata akurasi pada cross validation: 0.9096069868995634

Proses deployment:
- Load Model yang sudah disimpan
- Install Gradio
- Buat initerface untuk gradio nya
- siapkan requirements.txt
- unggah file yang dibutuhkan seperti notebooks, file app.py, requirements, datasetnya.




Link Web App nya : [Arisdwi/gradio-sms-classifier](https://huggingface.co/spaces/Arisdwi/gradio-sms-classifier)

![Screenshot 2023-12-28 182604](https://github.com/Arisdwi666/klasifikasi-sms-spam-with-gradio/assets/74097572/5b61e9c0-299f-417a-adbf-6b04c1f7470d)
