from flask import Flask, jsonify, send_from_directory, request
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from msrest.authentication import ApiKeyCredentials
from dotenv import load_dotenv
import os
from PIL import Image, ImageDraw
import numpy as np
import logging
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler()])

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/detect', methods=['POST'])
def detect_objects():
    try:
        load_dotenv()
        prediction_endpoint = os.getenv('PredictionEndpoint')
        prediction_key = os.getenv('PredictionKey')
        project_id = os.getenv('ProjectID')
        model_name = os.getenv('ModelName')

        if not all([prediction_endpoint, prediction_key, project_id, model_name]):
            logging.error("Environment variable missing or invalid")
            raise ValueError("One or more environment variables are missing or invalid.")

        logging.debug(f"Loaded env variables - Endpoint: {prediction_endpoint}, Key: {prediction_key}, ProjectID: {project_id}, ModelName: {model_name}")
        
        credentials = ApiKeyCredentials(in_headers={"Prediction-key": prediction_key})
        prediction_client = CustomVisionPredictionClient(prediction_endpoint, credentials)
        
        if 'image' not in request.files:
            logging.error("No file uploaded")
            return jsonify({'error': 'No file uploaded'}), 400
        
        image_file = request.files['image']
        if image_file.filename == '':
            logging.error("No file selected")
            return jsonify({'error': 'No file selected'}), 400

        input_file_name = 'input.jpg'
        output_file_name = 'output.jpg'

        logging.debug("Saving uploaded image")
        image_file.save(input_file_name)
        
        logging.debug("Opening image file")
        image = Image.open(input_file_name)
        h, w = image.size
        logging.debug(f"Image size: {image.size}")

        with open(input_file_name, mode="rb") as image_data:
            logging.debug("Sending request to prediction client")
            results = prediction_client.detect_image(project_id, model_name, image_data.read())
            logging.debug(f"Prediction results: {results}")

        image_with_boxes = process_results(image, results, h, w)
        image_with_boxes.save(output_file_name)

        pest_info = []
        if hasattr(results, 'predictions'):
            for prediction in results.predictions:
                if (prediction.probability * 100) > 50:
                    pest_info.append({
                        "name": prediction.tag_name,
                        "accuracy": f"{prediction.probability * 100:.2f}%"
                    })

        return jsonify({
            "message": "Detection complete",
            "output_file": f"/output/{output_file_name}",
            "pests": pest_info
        })
    except Exception as ex:
        logging.error("Exception occurred", exc_info=True)
        return jsonify({"error": str(ex)}), 500

@app.route('/output/<filename>')
def get_output_image(filename):
    return send_from_directory('.', filename)

def process_results(image, results, h, w):
    draw = ImageDraw.Draw(image)
    img_width, img_height = image.size

    if hasattr(results, 'predictions'):
        for prediction in results.predictions:
            if (prediction.probability * 100) > 50:
                left = prediction.bounding_box.left * img_width
                top = prediction.bounding_box.top * img_height
                width = prediction.bounding_box.width * img_width
                height = prediction.bounding_box.height * img_height
                points = [(left, top), (left + width, top), (left + width, top + height), (left, top + height)]
                draw.line(points + [points[0]], fill="magenta", width=int(img_width / 100))
                tag_name = f"{prediction.tag_name}: {prediction.probability * 100:.2f}%"
                draw.text((left, top), tag_name, fill="magenta")
    return image

if __name__ == "__main__":
    app.run(debug=True)
