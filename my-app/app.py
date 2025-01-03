from flask import Flask, render_template, request, send_file
from werkzeug.utils import secure_filename
import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import MeanSquaredError
from transformers import T5ForConditionalGeneration, T5Tokenizer
from gtts import gTTS
import speech_recognition as sr
from tensorflow.keras.preprocessing import image
import tensorflow as tf

# Disable GPU access programmatically
tf.config.set_visible_devices([], 'GPU')



app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['PROCESSED_FOLDER'] = 'processed'

# Load pre-trained models
emotion_model = load_model('cnn_model.h5', custom_objects={'mse': MeanSquaredError()})
t5_model = T5ForConditionalGeneration.from_pretrained("t5-base")
tokenizer = T5Tokenizer.from_pretrained("t5-base")

# Helper functions
def show_spectrogram(wav_path, save_path):
    y, sr = librosa.load(wav_path, sr=None)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    S_dB = librosa.power_to_db(S, ref=np.max)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def preprocess_spectrogram(spectrogram_path):
    img = image.load_img(spectrogram_path, color_mode="grayscale", target_size=(128, 128))
    img_array = image.img_to_array(img)  # Convert the image to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize if necessary
    return img_array

def predict_values(spectrogram_path):
    img_array = preprocess_spectrogram(spectrogram_path)
    print("Preprocessed image array shape:", img_array.shape)  # Should be (1, 128, 128, 1)

    predictions = emotion_model.predict(img_array)
    print("Raw predictions:", predictions)
    
    valence, arousal, dominance = predictions[0]
    return valence, arousal, dominance

def transcribe_audio(audio_file):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio = recognizer.record(source)
    try:
        return recognizer.recognize_google(audio)
    except:
        return ""

def apply_emotional_context(text, valence, arousal, dominance):
    if valence > 3:
        text = f"express positively: {text}"
    elif valence < 3:
        text = f"express sadly: {text}"
    else:
        text = f"express neutrally: {text}"
    if arousal > 3:
        text += "!!"
    elif arousal < 3:
        text += "."
    if dominance > 3:
        text = f"with confidence, {text}"
    elif dominance < 3:
        text = f"with hesitation, {text}"
    return text

def translate_and_generate_audio(text, output_filename):
    input_text = f"translate English to French: {text}"
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    output_ids = t5_model.generate(input_ids, max_length=40, num_beams=4, early_stopping=True)
    translated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    tts = gTTS(translated_text, lang='fr')
    tts.save(output_filename)
    return output_filename

# Routes
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        audio_file = request.files['audio_file']
        audio_filename = secure_filename(audio_file.filename)
        audio_path = os.path.join(app.config['UPLOAD_FOLDER'], audio_filename)
        audio_file.save(audio_path)

        spectrogram_path = os.path.join(app.config['PROCESSED_FOLDER'], 'spectrogram.png')
        show_spectrogram(audio_path, spectrogram_path)

        # Predict VAD
        valence, arousal, dominance = predict_values(spectrogram_path)
        print("Printing valence arousall--------------------------")
        print(valence, arousal, dominance)

        # Transcribe audio
        original_text = transcribe_audio(audio_path)
        if not original_text:
            return "Transcription failed.", 400

        # Apply emotional context
        modified_text = apply_emotional_context(original_text, valence, arousal, dominance)

        # Translate and generate audio
        output_audio_path = os.path.join(app.config['PROCESSED_FOLDER'], 'translated_audio.mp3')
        translate_and_generate_audio(modified_text, output_audio_path)

        return send_file(output_audio_path, as_attachment=True)

    return render_template('index.html')

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
