# PyTorch Emotion Tracker

Welcome to the PyTorch Emotion Tracker repository! This project focuses on real-time emotion detection using PyTorch and is equipped with tools to train new models.

## Getting Started

1. **Installation:** Install the required libraries using `pip install -r REQUIREMENTS.txt`.
2. **Running the Emotion Detector:** Execute `webcam.py` to start real-time emotion detection.
3. **Training a New Model:**
   - Download 'fer2013.tar.gz' from [this Kaggle link](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data?select=fer2013.tar.gz).
   - Move the downloaded file to `../datasets/raw/` in this repository.
   - Untar the file: `tar -xzf fer2013.tar`.
   - Modify the Network class in 'emotion_classifier.py' as needed.
   - Run 'checkpoints.py' to train the modified model.

## Acknowledgments

- [Keras VGGFace by rcmalli](https://github.com/rcmalli/keras-vggface)
- [Emotion Detection in Real Time by travistangvh](https://github.com/travistangvh/emotion-detection-in-real-time)
- Special thanks for the Dataset to the respective sources.

Enjoy exploring and utilizing the PyTorch Emotion Tracker!
