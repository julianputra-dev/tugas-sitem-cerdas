import threading
import csv
import os
from flask import Flask, request, jsonify, render_template
from core.preprocessing import TextPreprocessor
from core.naive_bayes import NaiveBayesClassifier
from train import execute_training
from deep_translator import GoogleTranslator

app = Flask(__name__)

preprocessor = TextPreprocessor()
model = NaiveBayesClassifier()
model.load_model('model/nb_weights.json')

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict_sentiment():
    payload = request.get_json()
    
    if not payload or 'review' not in payload:
        return jsonify({'error': 'Format muatan data tidak valid. Kunci review diperlukan.'}), 400
        
    raw_text = payload['review']
    
    # Translate text to English automatically before prediction
    try:
        translated_text = GoogleTranslator(source='auto', target='en').translate(raw_text)
        if not translated_text:
            translated_text = raw_text
    except Exception as e:
        print(f"Translation error: {e}")
        translated_text = raw_text
        
    processed_tokens = preprocessor.clean_text(translated_text)
    
    if not processed_tokens:
        return jsonify({'error': 'Teks tidak valid setelah pra-pemrosesan.'}), 400

    prediction = model.predict(processed_tokens)
    
    return jsonify({
        'status': 'success',
        # Send back the translated text so the UI uses English when sending feedback
        'input_text': translated_text,
        'predicted_class': prediction
    }), 200

def retrain_and_reload():
    try:
        print("Memulai retraining model dari umpan balik...")
        execute_training()
        global model
        model.load_model('model/nb_weights.json')
        print("Model berhasil diperbarui.")
    except Exception as e:
        print(f"Gagal melatih ulang model: {e}")

@app.route('/api/feedback', methods=['POST'])
def handle_feedback():
    payload = request.get_json()
    
    if not payload or 'review' not in payload or 'correction' not in payload:
        return jsonify({'error': 'Data umpan balik tidak lengkap.'}), 400
        
    review_text = payload['review']
    correction = payload['correction']
    
    try:
        file_path = 'data/steam_reviews.csv'
        with open(file_path, mode='a', encoding='utf-8', newline='') as file:
            fieldnames = ['date_posted', 'funny', 'helpful', 'hour_played', 'is_early_access_review', 'recommendation', 'review', 'title']
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writerow({
                'date_posted': '',
                'funny': '',
                'helpful': '',
                'hour_played': '',
                'is_early_access_review': '',
                'recommendation': correction,
                'review': review_text, # This is the translated english text
                'title': 'User Feedback'
            })
            
        threading.Thread(target=retrain_and_reload).start()
        
        return jsonify({'status': 'success', 'message': 'Umpan balik diterima dan AI sedang mempelajari ulang data.'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)