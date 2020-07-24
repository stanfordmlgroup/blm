from flask import Flask
import os

#from model import LSCCNN

basedir = os.path.abspath(os.path.dirname(__file__))

app = Flask(__name__)
app.config.from_object(__name__)

# Load model
#checkpoint_path = path_to_gcp_bucket
#model = LSCCNN(checkpoint_path=checkpoint_path)
#model.eval()
#model.cuda() ??

from app import views
