from facenet_pytorch import MTCNN
import matplotlib.pyplot as plt
import cv2
import time
import numpy as np
import torch
from model import Network
from device import DEVICE
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


emotion_dict = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Sad", 5: "Surprise", 6: 'Neutral'}


class Emotion_Recorder():
    def __init__(self,cam=1 , FPS=10, model=Network(), checkpoint_path="model.pt", show_webcam=False):
        self.cam = cam
        self.fps = FPS
        self.model = model
        self.checkpoint_path = checkpoint_path
        self.show_webcam = show_webcam

    def initialize_model(self):
        self.model_checkpoint = torch.load(self.checkpoint_path, map_location=torch.device(DEVICE))
        self.model.load_state_dict(self.model_checkpoint['net_state_dict'])
        self.model.eval() 

    def initialize_capture(self):
        self.mtcnn = MTCNN(keep_all=True)
        self.cap = cv2.VideoCapture(self.cam)

    def initialize_plot(self):
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.emotion_data = {'x': [], 'y': []}
        self.line, = self.ax.plot([], [], marker='o', linestyle='-', color='b')
        self.ax.set_xlabel('Real-time')
        self.ax.set_ylabel('Detected Emotion')
        self.ax.set_title('Real-time Emotion Plot')
    
    def initialize(self):
        self.initialize_model()
        self.initialize_capture()
        self.initialize_plot()

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return
        return frame

    def detect_face(self, frame):
        boxes, _ = self.mtcnn.detect(frame)
        if boxes is not None:
            box = [int(b) for b in boxes[0]]
            x1, y1, x2, y2 = box
            face = frame[y1:y2, x1:x2]

            if face.size != 0:
                face = cv2.resize(face, (96, 96))
                face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                face = np.expand_dims(face, axis=0)
                face = np.expand_dims(face, axis=0)
                face = torch.tensor(face, dtype=torch.float32)
                return face, box
            
    def detect_emotion(self, face):
        output = self.model(face)
        emotion = torch.argmax(output, dim=1).item()
        return emotion
    
    def plot_emotion(self, emotion):
        self.emotion_data['x'].append(len(self.emotion_data['x']))
        self.emotion_data['y'].append(emotion)
        self.line.set_xdata(self.emotion_data['x'])
        self.line.set_ydata(self.emotion_data['y'])
        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.flush_events()

    def start_recording(self):
        self.initialize()
        while True:
            frame = self.get_frame()
            try:
                face, box = self.detect_face(frame)
            except:
                continue
            if face is not None:
                emotion = self.detect_emotion(face)
                self.plot_emotion(emotion)
            if self.show_webcam:
                x1, y1, x2, y2 = box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
                cv2.putText(frame, emotion_dict[emotion], (x1 + 20, y1 - 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.imshow('Emotion Webcam', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            time.sleep(1/self.fps)
        self.cap.release()
        cv2.destroyAllWindows()






if __name__ == "__main__":
    r = Emotion_Recorder()
    r.start_recording()

        