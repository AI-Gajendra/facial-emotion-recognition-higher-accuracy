import cv2
import sqlite3
from datetime import datetime
from fer import FER

# Function to create the database and table if they don't exist
def create_database():
    with sqlite3.connect('emotion_entries.db') as conn:
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

# Function to insert data into the database
def insert_data(timestamp, expression, image_blob):
    with sqlite3.connect('emotion_entries.db') as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO entries (timestamp, expression, image) VALUES (?, ?, ?)
        ''', (timestamp, expression, image_blob))
        conn.commit()

def main():
    # Create the database
    create_database()

    # Initialize the FER model
    detector = FER()

    # Initialize the webcam
    webcam = cv2.VideoCapture(0)

    logged_faces = set()  # Set to keep track of logged faces

    while True:
        ret, frame = webcam.read()
        
        if not ret:
            print("Failed to grab frame")
            break

        # Detect emotions in the frame
        predictions = detector.detect_emotions(frame)

        for prediction in predictions:
            # Get bounding box and emotions
            box = prediction['box']
            emotions = prediction['emotions']
            main_emotion = max(emotions, key=emotions.get)

            # Prepare the image for storage
            x, y, w, h = box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (173, 216, 230), 2)
            cv2.putText(frame, f'Predicted: {main_emotion}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            # Prepare image for storage in database
            _, buffer = cv2.imencode('.jpg', frame[y:y + h, x:x + w])
            image_blob = buffer.tobytes()

            # Get current timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

            # Insert data into the database
            insert_data(timestamp, main_emotion, image_blob)

        cv2.imshow("Emotion Recognition", frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close windows
    webcam.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
