from flask import Flask, request, jsonify, send_from_directory
import cv2
import matplotlib.pyplot as plt
import random
import os
from flask_cors import CORS 

from service.imagePrediction import DeepFakeDetector
from service.videoPrediction import DeepFakeVideoDetector

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config["PROCESSED_FOLDER"] = "processed"

os.makedirs(app.config["PROCESSED_FOLDER"], exist_ok=True)


@app.route("/upload/file", methods=["POST"])
def upload_file_from_form():

    if "file" not in request.files:
        return jsonify({"message": "Nenhum arquivo enviado."}), 400

    file = request.files["file"]
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(file_path)

    task = request.form.get("description", "image")
    if task == "image":
        img = cv2.imread(file_path)
        if img is None:
            return (
                jsonify({"message": "ERROR 400: Não foi possível abrir a imagem."}),
                400,
            )

        # Usage example
        detector = DeepFakeDetector("model/xception_deepfake_image.h5")
        score, heatmap = detector.predict(file_path)
        classification = detector.classify(score)

        detector.plot_explanation(file_path, heatmap)
        detector.save_heatmap(file_path, heatmap, "path_to_save_heatmap.jpg")

        processed_image_path = os.path.join(
            app.config["PROCESSED_FOLDER"], file.filename
        )
        detector.save_heatmap(file_path, heatmap, processed_image_path)

        
        print(f"Score: {score}")
        print(f"Classification: {classification}")

        # return jsonify(
        #     {
        #         "score": score,
        #         "classification": classification,
        #         "heatmap_url": f"/processed/{file.filename}", 
        #     }
        # )

    elif task == "video":
        cap = cv2.VideoCapture(file_path)

        if not cap.isOpened():
            return (
                jsonify({"message": "ERROR 400: Não foi possível abrir o vídeo."}),
                400,
            )

        while True:
            ret, frame = cap.read()

            if not ret:
                break

        detector = DeepFakeVideoDetector("model/cuDNN_deepfake_video.h5")
        video_path = file_path
        score = detector.predict(video_path)
        classification = detector.classify(score)

    return jsonify(
        {
            "message": "Imagem exibida com sucesso.",
            "score": score,
            "target": classification,
        }
    )


@app.route("/processed/<filename>")
def get_processed_file(filename):
    return send_from_directory(app.config["PROCESSED_FOLDER"], filename)


if __name__ == "__main__":
    app.run(port=8086)
