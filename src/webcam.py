from facenet_pytorch import MTCNN
import cv2
import numpy as np
import torch
from models import Network
from device import DEVICE

# Load the MTCNN for face detection
mtcnn = MTCNN(keep_all=True)

# Load the emotion recognition model
model_checkpoint = torch.load('model.pt', map_location=torch.device(DEVICE))
model = Network()  # Initialize your model here
model.load_state_dict(model_checkpoint['net_state_dict'])
model.eval() 

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
        box = [int(b) for b in boxes[0]]
        x1, y1, x2, y2 = box
        face = frame[y1:y2, x1:x2]

        # Draw a rectangle around the face
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

        if face.size != 0:
            # Preprocess image for emotion recognition using the Network's forward pass
            face = cv2.resize(face, (96, 96))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            face = np.expand_dims(face, axis=0)
            face = np.expand_dims(face, axis=0)
            face = torch.tensor(face, dtype=torch.float32)

            # Get emotion prediction
            output = model(face)
            output = torch.argmax(output, dim=1).item()
            cv2.putText(frame, emotion_dict[output], (x1 + 20, y1 - 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)


    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
