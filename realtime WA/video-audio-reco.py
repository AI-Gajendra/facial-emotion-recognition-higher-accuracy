import cv2
import numpy as np
import sounddevice as sd
import speech_recognition as sr
import tensorflow as tf
from keras.models import model_from_json
import librosa
import librosa.display

import pickle

with open('scaler2.pickle', 'rb') as f:
    scaler2 = pickle.load(f)
    
with open('encoder2.pickle', 'rb') as f:
    encoder2 = pickle.load(f)



    


# Load facial emotion recognition model
def load_facial_model():
    json_file = open('facialemotionmodel.json', 'r')
    facial_model = model_from_json(json_file.read())
    json_file.close()
    facial_model.load_weights('facialemotionmodel.h5')
    return facial_model

# Load speech emotion recognition model
def load_speech_model():
    json_file = open('CNN_model.json', 'r')
    speech_model = model_from_json(json_file.read())
    json_file.close()
    speech_model.load_weights('best_model1_weights.h5')
    return speech_model

# Preprocess facial frame
def preprocess_facial_frame(frame):
    # Resize and preprocess frame as needed for your model
    # Example: resize to 48x48 pixels and normalize pixel values
    frame = cv2.resize(frame, (48, 48))
    frame = frame / 255.0
    frame = np.expand_dims(frame, axis=0)  # Add batch dimension
    return frame

# Preprocess audio data
def preprocess_audio(audio_data):
     # Convert audio data to a numpy array
    audio_array = np.frombuffer(audio_data.frame_data, dtype=np.int16)

    # Resample audio to a common sample rate if needed (e.g., 16 kHz)
    sample_rate = audio_data.sample_rate
    if sample_rate != 16000:
        audio_array = librosa.resample(audio_array, sample_rate, 16000)
        sample_rate = 16000

    # Extract MFCC features
    mfccs = librosa.feature.mfcc(y=audio_array, sr=sample_rate, n_mfcc=13)
    mfccs = np.mean(mfccs, axis=1)  # Take the mean of MFCC coefficients
    mfccs = np.expand_dims(mfccs, axis=0)  # Add batch dimension
    return mfccs

# Detect facial emotion
def detect_facial_emotion(facial_model, frame):
    preprocessed_frame = preprocess_facial_frame(frame)
    emotion_scores = facial_model.predict(preprocessed_frame)
    emotion_label = np.argmax(emotion_scores)
    return emotion_label

# Detect speech emotion
def detect_speech_emotion(speech_model, audio_data):
    preprocessed_audio = preprocess_audio(audio_data)
    emotion_scores = speech_model.predict(preprocessed_audio)
    emotion_label = np.argmax(emotion_scores)
    return emotion_label

# Main function
def main():
    # Load models
    facial_model = load_facial_model()
    speech_model = load_speech_model()

    # Capture video from webcam
    cap = cv2.VideoCapture(0)

    # Initialize speech recognition
    recognizer = sr.Recognizer()

    while True:
        # Capture a frame from the webcam
        ret, frame = cap.read()

        # Capture audio from the microphone
        with sr.Microphone() as source:
            print("Say something...")
            audio_data = recognizer.listen(source)

        # Detect facial emotion
        facial_emotion_label = detect_facial_emotion(facial_model, frame)

        # Detect speech emotion
        speech_emotion_label = detect_speech_emotion(speech_model, audio_data)

        # Print the detected emotions
        print(f"Facial Emotion: {facial_emotion_label}")
        print(f"Speech Emotion: {speech_emotion_label}")

        # Display the video frame with emotion labels
        cv2.imshow('Emotion Detection', frame)

        # Exit the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
