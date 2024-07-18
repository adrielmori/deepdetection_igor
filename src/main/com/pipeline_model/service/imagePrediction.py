import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import matplotlib.pyplot as plt
import cv2


class DeepFakeDetector:
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path)

    def preprocess_image(self, img_path):
        img = load_img(img_path, target_size=(299, 299))
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = tf.keras.applications.xception.preprocess_input(x)
        return x

    def predict(self, img_path):
        preprocessed_img = self.preprocess_image(img_path)
        prediction = self.model.predict(preprocessed_img)
        heatmap = self.generate_heatmap(preprocessed_img)
        return float(prediction[0][0]), heatmap

    def generate_heatmap(
        self, preprocessed_img, last_conv_layer_name="block14_sepconv2_act"
    ):
        grad_model = tf.keras.models.Model(
            [self.model.inputs],
            [self.model.get_layer(last_conv_layer_name).output, self.model.output],
        )

        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(preprocessed_img)
            preds = tf.convert_to_tensor(preds)
            class_channel_idx = int(tf.argmax(preds[0]))
            class_channel = preds[:, class_channel_idx]

        grads = tape.gradient(class_channel, last_conv_layer_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        heatmap = heatmap.numpy()

        return heatmap

    def classify(self, score, threshold=0.55):
        return "FAKE" if score > threshold else "REAL"

    def save_heatmap(self, img_path, heatmap, save_path, alpha=0.4):
        img = cv2.imread(img_path)
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        superimposed_img = heatmap * alpha + img
        cv2.imwrite(save_path, superimposed_img)

    def plot_explanation(self, img_path, heatmap, alpha=0.4):
        img = cv2.imread(img_path)
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        superimposed_img = heatmap * alpha + img
        plt.imshow(superimposed_img[:, :, ::-1] / 255)
        plt.axis("off")
