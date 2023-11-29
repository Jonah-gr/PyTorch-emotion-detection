# PyTorch Emotion Tracker

Welcome to the PyTorch Emotion Tracker repository! This project focuses on real-time emotion detection using PyTorch and is equipped with tools to train new models.

## Getting Started

1. **Clone the repository:** Clone this repository using `git clone https://github.com/Jonah-gr/PyTorch-emotion-detection.git`.
2. **Installation:** Install the required libraries using `pip install -r REQUIREMENTS.txt`.
3. **Running the Emotion Detector:** 
   - Download [model.pt](https://1drv.ms/u/s!AtugCP0Mx48phOwKPvDIkmIC81-6oA?e=47oA9N).
   - Move `model.pt` to `../src/`.
   - Execute `webcam.py` to start real-time emotion detection.

4. **Training a New Model:**
   - Download [fer2013.tar.gz](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data?select=fer2013.tar.gz).
   - Move the downloaded file to `../datasets/raw/` in this repository.
   - Untar the file: `tar -xzf fer2013.tar`.
   - Modify the Network class in `model.py` as needed.
   - Run `checkpoints.py` to train the modified model.

## Acknowledgments

- [Keras VGGFace by rcmalli](https://github.com/rcmalli/keras-vggface)
- [Emotion Detection in Real Time by travistangvh](https://github.com/travistangvh/emotion-detection-in-real-time)
- Special thanks for the Dataset to the respective sources.

Enjoy exploring and utilizing the PyTorch Emotion Tracker!
