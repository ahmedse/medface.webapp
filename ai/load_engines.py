# models.py
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.models import load_model
from mtcnn import MTCNN
from django.conf import settings
from django.templatetags.static import static
import pickle
from contextlib import contextmanager
from functools import wraps
import sys
import io

def capture_output(func):
    """Wrapper to capture print output."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        old_stdout = sys.stdout
        new_stdout = io.StringIO()
        sys.stdout = new_stdout
        try:
            return func(*args, **kwargs)
        finally:
            sys.stdout = old_stdout
    return wrapper

detector = MTCNN()
w_detect_face = capture_output(detector.detect_faces) 
# ai models
resnet50_model = load_model(os.path.join(settings.AI_ROOT, 'resnet50.h5'))
senet50_model = load_model(os.path.join(settings.AI_ROOT, 'senet50.h5'))
vgg16_model = load_model(os.path.join(settings.AI_ROOT, 'vgg16.h5'))
# Load class labels
with open(os.path.join(settings.AI_ROOT, 'face-labels.pickle'), 'rb') as f:
    class_labels = pickle.load(f)

# photo enhancer model
import cv2
sr_model = cv2.dnn_superres.DnnSuperResImpl_create()
path = os.path.join(settings.AI_ROOT, 'EDSR_x3.pb') 
sr_model.readModel(path)
sr_model.setModel("edsr", 3)
