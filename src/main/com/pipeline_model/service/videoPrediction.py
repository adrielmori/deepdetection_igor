import keras
import numpy as np
import imageio
import cv2
from tensorflow.keras.preprocessing.image import load_img, img_to_array

IMG_SIZE = 224
BATCH_SIZE = 64
EPOCHS = 10

MAX_SEQ_LENGTH = 20
NUM_FEATURES = 2048

def build_feature_extractor():
    feature_extractor = keras.applications.InceptionV3(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    preprocess_input = keras.applications.inception_v3.preprocess_input

    inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3))
    preprocessed = preprocess_input(inputs)

    outputs = feature_extractor(preprocessed)
    return keras.Model(inputs, outputs, name="feature_extractor")


feature_extractor = build_feature_extractor()

class DeepFakeVideoDetector:
    def __init__(self, model_path):
        self.model = keras.models.load_model(model_path)

    def load_video(self, path):
        cap = cv2.VideoCapture(path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        frames = np.asarray(frames)
        return frames

    def prepare_single_video(self, frames):
        frames = frames[None, ...]
        frame_mask = np.zeros(shape=(1, MAX_SEQ_LENGTH,), dtype="bool")
        frame_features = np.zeros(shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32")

        for i, batch in enumerate(frames):
            video_length = batch.shape[0]
            length = min(MAX_SEQ_LENGTH, video_length)
            for j in range(length):
                resized_frame = cv2.resize(batch[j, :], (224, 224))  # Redimensiona o frame para (224, 224)
                frame_features[i, j, :] = feature_extractor.predict(resized_frame[None, :])
            frame_mask[i, :length] = 1 

        return frame_features, frame_mask


    def predict(self, video_path):
        frames = self.load_video(video_path)
        frame_features, frame_mask = self.prepare_single_video(frames)
        prediction = self.model.predict([frame_features, frame_mask])
        return float(prediction[0][0])

    def classify(self, score, threshold=0.62):
        return "FAKE" if score > threshold else "REAL"