from flask import Flask, render_template, request
from model import LSCCNN
from PIL import Image
from io import BytesIO

import logging
import numpy
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

    MODEL = LSCCNN(checkpoint_path=MODEL_FILENAME)
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


@app.route("/model", methods=['POST'])
def model():
    if request.method == 'POST':
        logging.info("Request received, loading image ...")
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
        logging.info("Image loaded, image size: " + str(image_file_size))

        if image_file_size > 5000000:
            with BytesIO() as f:
                img_size_test.save(f, "JPEG")
                image = f.getvalue()

                image = Image.open(BytesIO(image))
                image = numpy.array(image)

        jpg_txt = run_inference(image)

        return jpg_txt, 200