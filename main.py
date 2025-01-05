import cv2
from mtcnn import MTCNN
from deepface import DeepFace
import numpy as np

# Initialize the MTCNN face detector
detector = MTCNN()

# Start the webcam capture
cap = cv2.VideoCapture(0)  # 0 for default webcam, or specify the webcam index
cap.set(3, 800)  # Set width
cap.set(4, 600)  # Set height

# Emotion buffer for stability
emotion_buffer = []
buffer_size = 5  # Number of frames to stabilize the emotion

# Frame skip for optimization
frame_skip = 3  # Process every 3rd frame
frame_count = 0
stable_emotion = "Neutral"  # Default emotion
faces = []  # Initialize faces

while True:
    # Read frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Mirror the frame
    frame = cv2.flip(frame, 1)

    # Increment frame count
    frame_count += 1

    # Convert frame to RGB for processing
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process only every `frame_skip` frames
    if frame_count % frame_skip == 0:
        faces = detector.detect_faces(frame_rgb)

        if faces:
            # Select the face with the highest confidence
            faces = sorted(faces, key=lambda x: x['confidence'], reverse=True)
            x, y, w, h = faces[0]['box']

            try:
                # Resize the face region for faster emotion analysis
                face_region = frame_rgb[y:y + h, x:x + w]
                face_resized = cv2.resize(face_region, (48, 48))

                # Analyze emotions using DeepFace
                result = DeepFace.analyze(face_resized, actions=['emotion'], enforce_detection=False)

                # Get dominant emotion
                dominant_emotion = max(result[0]['emotion'], key=result[0]['emotion'].get)

                # Add dominant emotion to buffer
                emotion_buffer.append(dominant_emotion)
                if len(emotion_buffer) > buffer_size:
                    emotion_buffer.pop(0)

                # Find the most frequent emotion in the buffer
                stable_emotion = max(set(emotion_buffer), key=emotion_buffer.count)

            except Exception as e:
                print(f"Error: {e}")

    # Continuously display the bounding box and emotion
    if faces:
        x, y, w, h = faces[0]['box']
        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # Display the stable dominant emotion above the head
        cv2.putText(frame, stable_emotion, 
                    (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

    # Display the live camera feed
    cv2.imshow("Smooth Real-Time Emotion Detection (Mirrored)", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
