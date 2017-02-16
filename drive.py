# 1- Function Set #########################################
import socketio
import skimage.transform as sktransform
from PIL import Image
from io import BytesIO
import base64
import numpy as np

sio = socketio.Server()

### Use preprocess function to resize image
def preprocess(image, top_offset=.375, bottom_offset=.125):
    top = int(top_offset * image.shape[0])
    bottom = int(bottom_offset * image.shape[0])
    image = sktransform.resize(image[top:-bottom, :], (32, 128, 3))
    return image

@sio.on('telemetry')
def telemetry(sid, data):
### The current steering angle of the car
    steering_angle = data["steering_angle"]
### The current throttle of the car
    throttle = data["throttle"]
### The current speed of the car
    speed = data["speed"]
### The current image from the center camera of the car
    imgString = data["image"]
    image = Image.open(BytesIO(base64.b64decode(imgString)))
    image_array = preprocess(np.asarray(image))
    steering_angle = float(model.predict(image_array[None, :, :, :], batch_size=1))
    throttle = .01 if abs(steering_angle) > 0.1 and float(speed) > 10 else .15  # else .5 during challenge model
    print(steering_angle, throttle)
    send_control(steering_angle, throttle)

@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)

def send_control(steering_angle, throttle):
    sio.emit("steer", data={'steering_angle': steering_angle.__str__(), 'throttle': throttle.__str__()}, skip_sid=True)

# 2-Process ###############################################
import argparse
from keras.models import load_model
from flask import Flask
import eventlet
import eventlet.wsgi

model = None
app = Flask(__name__)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('model', type=str, help='Path to model h5 file. Model should be on the same path.')
    args = parser.parse_args()
    model = load_model(args.model)
    app = socketio.Middleware(sio, app)
### Deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)