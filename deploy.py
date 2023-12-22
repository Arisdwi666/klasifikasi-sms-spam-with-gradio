import joblib
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from preprocessing_functions import casefolding, text_normalize, remove_stop_words, stemming
from preprocessing_functions import text_preprocessing_process
import gradio as gr


# Load the TF-IDF vectorizer using joblib
with open('model/kbest_feature.pickle', 'rb') as f:
    vocab  = pickle.load(f)

# Load the Naive Bayes model using pickle
with open('model/model_1.joblib', 'rb') as f:
    model = joblib.load(f)

def predict_klasifikasi(text):
    pre_input_text = text_preprocessing_process(text)
    # Transform the new text data using the loaded TF-IDF vectorizer
    tf_idf_vec = TfidfVectorizer(vocabulary=set(vocab)) 
    pred = model.predict(tf_idf_vec.fit_transform([pre_input_text]))  # Lakukan prediksi
    
    if pred == 0:
        return 'SMS Normal'
    elif pred == 1:
        return 'SMS Fraud'
    else:
        return 'SMS Promo'

# Create a Gradio interface
iface = gr.Interface(fn=predict_klasifikasi, inputs="text", outputs="label", title="Klasifikasi SMS", description="Klasifikasi SMS menggunakan model Naive Bayes")

# Launch the interface
iface.launch(share=True)

