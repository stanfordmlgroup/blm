from flask import Flask, render_template, request
from model import LSCCNN
from PIL import Image
from io import BytesIO

import logging
import numpy as np
import base64
import cv2

MODEL_FILENAME = './model/scale_4_epoch_46.pth'
MODEL = None

EMOJI_FILENAME = './blm_fist.png'
EMOJI = None

app = Flask(__name__)
    
@app.before_first_request
def _load_model():
    global MODEL
    global EMOJI

    logging.basicConfig(level=logging.INFO)
    logging.info("Loading model '" + MODEL_FILENAME + "' ...")

    MODEL = LSCCNN()
    MODEL.load_weights(MODEL_FILENAME)
    MODEL.eval()

    EMOJI = cv2.imread(EMOJI_FILENAME, -1)

    logging.info("Model loaded")


@app.route("/")
def index():
    return render_template("index.html")


def run_inference(image):
    global MODEL
    global EMOJI
    logging.info("Inference starting ...")
    _, _, img_out = MODEL.predict_single_image(image, EMOJI, nms_thresh=0.25)
    logging.info("Inference completed")
    img_out = cv2.cvtColor(img_out, cv2.COLOR_RGB2BGR)
    ret, buf = cv2.imencode('.jpg', img_out)

    jpg_text = base64.b64encode(buf)

    return jpg_text


def read_image(fs):
    logging.info("Reading image file {} ...".format(fs))
    image = Image.open(fs)
    logging.info("Image file read: {}".format(image))

    return np.array(image)


def handle_alpha_channel(image):
    if image.shape[2] > 3:
        image = image[:, :, :3]
        logging.info("Alpha channel removed")
    
    return image


@app.route("/model", methods=['POST'])
def model():
    if request.method == 'POST':
        file_storage = request.files["inputimage"]
        image = read_image(file_storage)
        image = handle_alpha_channel(image)

        jpg_txt = run_inference(image)

        return jpg_txt, 200

if __name__ == '__main__':
    app.run("0.0.0.0", debug=True)