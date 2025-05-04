from flask import Flask, request, jsonify
from flask_cors import CORS 
from util.inference_utils import inference, create_model
import cv2
import base64
import numpy as np
from io import BytesIO
from PIL import Image
import os
import logging

app = Flask(__name__)

cors_origins = os.getenv("CORS_ORIGINS", "*")
CORS(app, resources={r"/predict": {"origins": cors_origins}})

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
class Parameters:
    def __init__(self):
        self.isTrain = False
        self.input_nc = 3
        self.output_nc = 1
        self.ngf = 64
        self.netG = 'deeplabv3plus_mobilenet'
        self.norm = 'batch'
        self.no_dropout = True
        self.init_type = 'normal'
        self.init_gain = 0.02
        self.gpu_ids = []  
        self.num_classes = 1

opt = Parameters()
try:
    model = create_model(opt, cp_path='pretrained_net_G.pth')
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    raise

@app.route("/predict", methods=["POST"])
def predict():
    # image is FormData
    if "image" not in request.files:
        logger.warning("No image file provided in request")
        return jsonify({"error": "No image file provided"}), 400

    image_file = request.files["image"]
    
    # image type
    if not image_file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        logger.warning(f"Unsupported file type: {image_file.filename}")
        return jsonify({"error": "Unsupported file type. Use PNG, JPG, or JPEG."}), 400

    image_file.seek(0, os.SEEK_END)
    file_size = image_file.tell()
    if file_size > 10 * 1024 * 1024:
        logger.warning(f"Image file too large: {file_size} bytes")
        return jsonify({"error": "Image file too large. Max size is 10MB."}), 400
    image_file.seek(0)
    try:
        width = int(request.form.get("width", 256))
        height = int(request.form.get("height", 256))
        unit = request.form.get("unit", "px")
        if width <= 0 or height <= 0:
            raise ValueError("Width and height must be positive")
    except ValueError as e:
        logger.warning(f"Invalid parameters: {str(e)}")
        return jsonify({"error": f"Invalid parameters: {str(e)}"}), 400


    try:
        image_bytes = image_file.read()
        img = Image.open(BytesIO(image_bytes))
        img.verify() 
        logger.info(f"Received image: {image_file.filename}, size: {file_size} bytes")
    except Exception as e:
        logger.error(f"Invalid image file: {str(e)}")
        return jsonify({"error": f"Invalid image file: {str(e)}"}), 400


    try:
        result_img, _ = inference(model, image_bytes, (width, height), unit)
        _, buffer = cv2.imencode(".jpg", result_img)
        result_base64 = base64.b64encode(buffer).decode("utf-8")
        logger.info("Inference completed successfully")
        return jsonify({
            "result_image": result_base64,
            "status": "success"
        }), 200
    except Exception as e:
        logger.error(f"Inference failed: {str(e)}")
        return jsonify({"error": f"Inference failed: {str(e)}"}), 500
if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)