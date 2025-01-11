import cv2
from keras.models import model_from_json
import numpy as np
import sqlite3
from datetime import datetime
import os

# Function to create the database and table if they don't exist
def create_database():
    conn = sqlite3.connect('lab_entries.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS entries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            expression TEXT NOT NULL,
            image BLOB NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

# Function to insert data into the database
def insert_data(timestamp, expression, image_blob):
    conn = sqlite3.connect('lab_entries.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO entries (timestamp, expression, image) VALUES (?, ?, ?)
    ''', (timestamp, expression, image_blob))
    conn.commit()
    conn.close()

def main():
    # Create the database
    create_database()

    # Load the pre-trained model
    json_file = open("facialemotionmodel.json", "r")
    model_json = json_file.read()
    json_file.close()
    model = model_from_json(model_json)
    model.load_weights("facialemotionmodel.h5")

    # Load Haar cascade for face detection
    haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(haar_file)

    def extract_features(image):
        feature = np.array(image)
        feature = feature.reshape(1, 48, 48, 1)
        return feature / 255.0

    # Initialize the webcam
    webcam = cv2.VideoCapture(0)
    labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

    logged_faces = set()  # Set to keep track of logged faces

    while True:
        ret, im = webcam.read()
        
        # Check if the frame is captured successfully
        if not ret:
            print("Failed to grab frame")
            continue

        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (p, q, r, s) in faces:
            face_id = (p, q, r, s)

            # If this face has not been logged yet
            if face_id not in logged_faces:
                image = gray[q:q+s, p:p+r]
                cv2.rectangle(im, (p, q), (p+r, q+s), (173, 216, 230), 2)
                image = cv2.resize(image, (48, 48))
                img = extract_features(image)
                pred = model.predict(img)
                prediction_label = labels[pred.argmax()]

                # Prepare image for storage in database
                _, buffer = cv2.imencode('.jpg', im[q:q+s, p:p+r])
                image_blob = buffer.tobytes()  # Convert to binary

                # Get current timestamp
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

                # Insert data into the database
                insert_data(timestamp, prediction_label, image_blob)

                # Mark this face as logged
                logged_faces.add(face_id)

                # Draw the prediction text on the image
                cv2.putText(im, f'Predicted: {prediction_label}', (p, q - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        cv2.imshow("Output", im)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close windows
    webcam.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
