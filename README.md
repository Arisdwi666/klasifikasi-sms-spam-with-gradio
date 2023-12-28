 Klasifikasi SMS Spam Berbahasa Indonesia Menggunakan Metode Multinomial Naive Bayes & Feature Selection Chi-Square dan Deploy menggunakan Gradio - Hosting menggunakan Hugging Face

Dataset diambil dari github https://github.com/ksnugroho/klasifikasi-spam-sms/ dengan jumlah data 1143 data. 569 data untuk SMS Normal, 335 data untuk Peniuan/Fraud, 239 data untuk Promo. 

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

Link Web App nya : [Arisdwi/gradio-sms-classifier](https://huggingface.co/spaces/Arisdwi/gradio-sms-classifier)

![Screenshot 2023-12-28 182604](https://github.com/Arisdwi666/klasifikasi-sms-spam-with-gradio/assets/74097572/5b61e9c0-299f-417a-adbf-6b04c1f7470d)
