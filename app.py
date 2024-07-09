import streamlit as st
import torch
import torchaudio
from sklearn.preprocessing import StandardScaler
import pickle
import librosa
import numpy as np

# Load scaler
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Function to extract features from an audio file
def extract_features(file_path, offset):
    try:
        y, sr = librosa.load(file_path, sr=None, offset=offset, duration=5.0)  # Load audio with offset
    except Exception as e:
        st.error(f"Error loading {file_path}: {e}")
        return None

    features = {}

    features['length'] = len(y) / sr
    features['chroma_stft_mean'] = np.mean(librosa.feature.chroma_stft(y=y, sr=sr))
    features['chroma_stft_var'] = np.var(librosa.feature.chroma_stft(y=y, sr=sr))
    features['rms_mean'] = np.mean(librosa.feature.rms(y=y))
    features['rms_var'] = np.var(librosa.feature.rms(y=y))
    features['spectral_centroid_mean'] = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    features['spectral_centroid_var'] = np.var(librosa.feature.spectral_centroid(y=y, sr=sr))
    features['spectral_bandwidth_mean'] = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    features['spectral_bandwidth_var'] = np.var(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    features['rolloff_mean'] = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    features['rolloff_var'] = np.var(librosa.feature.spectral_rolloff(y=y, sr=sr))
    features['zero_crossing_rate_mean'] = np.mean(librosa.feature.zero_crossing_rate(y))
    features['zero_crossing_rate_var'] = np.var(librosa.feature.zero_crossing_rate(y))
    features['harmony_mean'] = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr))
    features['harmony_var'] = np.var(librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr))
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    features['perceptr_mean'] = np.mean(mfccs)
    features['perceptr_var'] = np.var(mfccs)
    features['tempo'] = librosa.feature.rhythm.tempogram(y=y, sr=sr)[0, 0]

    for i in range(1, 21):
        features[f'mfcc{i}_mean'] = np.mean(mfccs[i-1])
        features[f'mfcc{i}_var'] = np.var(mfccs[i-1])

    return features

# Streamlit UI
st.title('Music Genre Classification')

uploaded_file = st.file_uploader("Choose an audio file...", type=["wav", "mp3"])

if uploaded_file:
    st.audio(uploaded_file, format='audio/wav')

    # Extract features from the uploaded audio file
    features = extract_features(uploaded_file.name, 0)

    if features:
        # Prepare features for scaling
        features_for_scaling = {k: v for k, v in features.items() if k != 'filename'}

        # Normalize features
        features_arr = np.array([list(features_for_scaling.values())])
        features_scaled = scaler.transform(features_arr)
        features_tensor = torch.tensor(features_scaled, dtype=torch.float32)

        # Path to the trained model
        best_model_path = "C:/Users/gus ryan/iDev/music-classification-models/the_best_model.pkl"

        # Load the trained model
        with open(best_model_path, 'rb') as file:
            model = pickle.load(file)

        # Make prediction
        with torch.no_grad():
            outputs = model(torch.unsqueeze(torch.tensor(features_scaled), 1).to('cuda' if torch.cuda.is_available() else 'cpu'))
            _, predicted = torch.max(outputs, 1)

        # Mapping antara label numerik dan nama genre
        genre_mapping = {
            0: 'Blues',
            1: 'Classical',
            2: 'Country',
            3: 'Disco',
            4: 'Hiphop',
            5: 'Jazz',
            6: 'Metal',
            7: 'Pop',
            8: 'Reggae',
            9: 'Rock'
        }

        # Get the predicted label
        predicted_label = predicted.item()

        # Konversi nilai numerik menjadi nama genre menggunakan mapping
        predicted_genre = genre_mapping[predicted_label]

        st.success(f'Predicted Genre: {predicted_genre}')