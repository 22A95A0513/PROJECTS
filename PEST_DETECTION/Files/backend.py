from flask import Flask, jsonify, send_from_directory, request
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from msrest.authentication import ApiKeyCredentials
from dotenv import load_dotenv
import os
from PIL import Image, ImageDraw
import numpy as np

app = Flask(__name__)

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
            raise ValueError("One or more environment variables are missing or invalid.")

        credentials = ApiKeyCredentials(in_headers={"Prediction-key": prediction_key})
        prediction_client = CustomVisionPredictionClient(prediction_endpoint, credentials)

        if 'image' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Overwrite input.jpg and output.jpg
        input_file_name = 'input.jpg'
        output_file_name = 'output.jpg'

        # Save the uploaded image to input.jpg
        image_file.save(input_file_name)

        # Open the image for processing
        image = Image.open(input_file_name)
        h, w, ch = np.array(image).shape

        with open(input_file_name, mode="rb") as image_data:
            results = prediction_client.detect_image(project_id, model_name, image_data)

        image_with_boxes = process_results(image, results, h, w)

        # Save the processed image as output.jpg
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
        return jsonify({"error": str(ex)}), 500

@app.route('/output/<filename>')
def get_output_image(filename):
    return send_from_directory('.', filename)

def process_results(image, results, h, w):
    draw = ImageDraw.Draw(image)
    if hasattr(results, 'predictions'):
        for prediction in results.predictions:
            if (prediction.probability * 100) > 50:
                left = prediction.bounding_box.left * w
                top = prediction.bounding_box.top * h
                width = prediction.bounding_box.width * w
                height = prediction.bounding_box.height * h

                points = [(left, top), (left + width, top), (left + width, top + height), (left, top + height)]
                draw.line(points + [points[0]], fill="magenta", width=int(w / 100))

                tag_name = f"{prediction.tag_name}: {prediction.probability * 100:.2f}%"
                draw.text((left, top), tag_name, fill="magenta")

    return image

if __name__ == "__main__":
    app.run(debug=True)
