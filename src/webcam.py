from facenet_pytorch import MTCNN
import cv2
import numpy as np
import torch

FPS = 60

# Load the MTCNN for face detection
mtcnn = MTCNN(keep_all=True)

# Load the emotion recognition model
model = torch.load('model.pt')  # Load your emotion detection PyTorch model here
# print(model)
# model.eval()

# Dictionary which assigns each label an emotion (alphabetical order)
emotion_dict = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Sad", 5: "Surprise", 6: 'Neutral'}

# Start the webcam feed
cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect faces in the image
    boxes, _ = mtcnn.detect(frame)
    if boxes is not None:
        for box in boxes:
            box = [int(b) for b in box]
            x1, y1, x2, y2 = box
            face = frame[y1:y2, x1:x2]

            # Draw a rectangle around the face
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # Preprocess image for emotion recognition
            face = cv2.resize(face, (48, 48))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            face = np.expand_dims(face, axis=0)
            face = np.expand_dims(face, axis=0)
            face = torch.tensor(face, dtype=torch.float32)

            # Get emotion prediction
            output = model(face).detach().numpy()
            maxindex = int(np.argmax(output))
            cv2.putText(frame, emotion_dict[maxindex], (x1 + 20, y1 - 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
