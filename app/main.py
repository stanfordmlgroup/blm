from flask import Flask, render_template, jsonify, request, send_file
from google.cloud import storage
from model import LSCCNN
from PIL import Image
from io import BytesIO

import urllib.request
import logging
import torch
import numpy
import base64
import json
# import uuid
import os
import cv2

import google.cloud.logging

# Instantiates a client
client = google.cloud.logging.Client()

# Retrieves a Cloud Logging handler based on the environment
# you're running in and integrates the handler with the
# Python logging module. By default this captures all logs
# at INFO level and higher
client.get_default_handler()
client.setup_logging()


MODEL_BUCKET = os.environ['MODEL_BUCKET']
MODEL_FILENAME = os.environ['MODEL_FILENAME']
MODEL = None
EMOJI = None

app = Flask(__name__)


@app.before_first_request
def _load_model():
    global MODEL
    global EMOJI
    client = storage.Client()
    bucket = client.get_bucket(MODEL_BUCKET)
    blob = bucket.get_blob(MODEL_FILENAME)
    weights = blob.download_to_filename("model_weights.pth")

    MODEL = LSCCNN(checkpoint_path="model_weights.pth")
    # MODEL.cuda()
    MODEL.eval()

    EMOJI = cv2.imread("blm_fist.png", -1)
    logging.info("Model loaded")


@app.route("/")
def index():
    return render_template("index.html")


def run_inference(image):
    global MODEL
    global EMOJI
    logging.info("Inference starting")
    _, _, img_out = MODEL.predict_single_image(image, EMOJI, nms_thresh=0.25)
    logging.info("Inference Completed")
    img_out = cv2.cvtColor(img_out, cv2.COLOR_RGB2BGR)
    ret, buf = cv2.imencode('.jpg', img_out)

    jpg_text = base64.b64encode(buf)


    return jpg_text


@app.route("/model", methods=['POST'])
def model():
    if request.method == 'POST':
        logging.info("Request Received")
        image = request.files["inputimage"]
        image.seek(0)
        contents = image.read()
        image = Image.open(BytesIO(contents))
        image = numpy.array(image) # RGB (so no need to do BGR2RGB)

        # remove alpha channel from png
        if image.shape[2] > 3:
            image = image[:, :, :3]

        # determine image size
        img_file = BytesIO()
        img_size_test = Image.fromarray(image.copy())
        img_size_test.save(img_file, 'png')
        image_file_size = img_file.tell()
        logging.info("Image size " + str(image_file_size))

        if image_file_size > 1000000:
            with BytesIO() as f:
                img_size_test.save(f, "JPEG")
                image = f.getvalue()

                image = Image.open(BytesIO(image))
                image = numpy.array(image)


        jpg_txt = run_inference(image)

        return jpg_txt, 200


if __name__ == '__main__':
	port = int(os.environ.get("PORT", 5000))
	app.run(host='127.0.0.1', port=8000, debug=True)
