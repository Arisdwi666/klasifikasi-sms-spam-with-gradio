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

with gr.Blocks(title="SMS classification with Multinomial Naive Bayes",
            css=".gradio-container {background:pink ;}"
        ) as demo:
    gr.HTML(
        """
        <div style="text-align: center; max-width: 1600px; margin: 20px auto;">
        <h2 style="font-weight: 900; font-size: 2.3rem; margin: 0rem; color: black">
            SMS-Classifier: "Sort the Text, Shield from Spam."
        </h2>
        <h2 style="font-weight: 460; font-size: 1.1rem; margin: 0rem">
            <a href="https://github.com/Arisdwi666/klasifikasi-sms-spam-with-gradio">Aris Dwi</a>
        </h2>      
        <h2 style="font-weight: 460; font-size: 1.1rem; margin: 0rem">
            Teknik Informatika, Universitas Dian Nuswantoro, STKI
        </h2>
        <h2 style="text-align: left; font-weight: 450; font-size: 1rem; margin-top: 0.5rem; margin-bottom: 0.5rem">
        We propose <b>SMS Classifier</b>, for Discover the safety and convenience of communicating with SMS through our web app!. By using <b>Multinomial Naive Bayes method and Chi-Square feature selection</b> we provide an intelligent Indonesian spam SMS classification solution. Easy to use through Gradio's intuitive interface, our platform provides not only security, but also convenience in filtering your messages.</b> Explore now for a safer and more efficient communication experience!
        </h2>
        """)

    with gr.Row(): 
        input_text = gr.Text(type="text", label="Input Text") 
        output_label=gr.Label(label="Output Label")

    with gr.Row():
        send_btn = gr.Button("Submit")
        clr_btn = gr.ClearButton(input_text)

    send_btn.click(fn=predict_klasifikasi, inputs=input_text, outputs=output_label)

demo.launch()

