### 1- Function Set #######################################
import skimage.transform as sktransform
import numpy as np
import matplotlib.image as mpimg
import os
import random
from keras.callbacks import Callback
import shutil

def preprocess(image, top_offset=.375, bottom_offset=.125):
    top = int(top_offset * image.shape[0])
    bottom = int(bottom_offset * image.shape[0])
    image = sktransform.resize(image[top:-bottom, :], (32, 128, 3))
    return image

### Keras generator yielding batches of training/validation data if augment=True
def generator(data, augment=True):
    while True:
    ### Generate random batch of indices
        indices = np.random.permutation(data.count()[0])
        batch_size = 128
        for batch in range(0, len(indices), batch_size):
            batch_indices = indices[batch:(batch + batch_size)]
        ### Output arrays
            x = np.empty([0, 32, 128, 3], dtype=np.float32)
            y = np.empty([0], dtype=np.float32)
        ### Read in and preprocess a batch of images
            for i in batch_indices:
            ### Randomly select camera
                camera = np.random.randint(len(cameras)) if augment else 1
            ### Read frame image and work out steering angle
                image = mpimg.imread(os.path.join(data[cameras[camera]].values[i].strip()))
                angle = data.steering.values[i] + cameras_steering_correction[camera]
                if augment:
                ### Add random shadow as a vertical slice of image
                    h, w = image.shape[0], image.shape[1]
                    [x1, x2] = np.random.choice(w, 2, replace=False)
                    k = h / (x2 - x1)
                    b = - k * x1
                    for i in range(h):
                        c = int((i - b) / k)
                        image[i, :c, :] = (image[i, :c, :] * .5).astype(np.int32)
			### Randomly shift up and down, crops 'top_offset' & 'bottom_offset', resizes to 32x128 px and scales pixel values to [0, 1]			
                v_delta = .05 if augment else 0
                image = preprocess(image,
				                   top_offset=random.uniform(.375 - v_delta, .375 + v_delta),
								   bottom_offset=random.uniform(.125 - v_delta, .125 + v_delta))
            ### Append to batch
                x = np.append(x, [image], axis=0)
                y = np.append(y, [angle])
        ### Randomly flip half of images in the batch
            flip_indices = random.sample(range(x.shape[0]), int(x.shape[0] / 2))
            x[flip_indices] = x[flip_indices, :, ::-1, :]
            y[flip_indices] = -y[flip_indices]
            yield (x, y)
			
# Callbacks function in model, use for save best model in each epoch, but it is not neccesary
class weight_logger(Callback):
    def __init__(self):
        super(weight_logger, self).__init__()
        self.weight_path = os.path.join('weights/')
        shutil.rmtree(self.weight_path, ignore_errors=True)
        os.makedirs(self.weight_path, exist_ok=True)
    def on_epoch_end(self, epoch, logs={}):
        self.model.save_weights(os.path.join(self.weight_path, 'model_epoch_{}.h5'.format(epoch + 1)))
		
### 2- Process ############################################
import pandas as pd
from sklearn import model_selection
from keras import models, optimizers
from keras.layers import convolutional, Lambda, ELU, pooling, core

### Load and split data
df = pd.io.parsers.read_csv('driving_log.csv')
train_data, valid_data = model_selection.train_test_split(df, test_size=.2)

### Cameras setting
cameras = ['left', 'center', 'right']
cameras_steering_correction = [.25, 0., -.25]
			
### Train Model
model = models.Sequential()
model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(32, 128, 3)))
model.add(convolutional.Convolution2D(16, 3, 3, input_shape=(32, 128, 3)))
model.add(ELU())
model.add(pooling.MaxPooling2D(pool_size=(2, 2)))
model.add(convolutional.Convolution2D(32, 3, 3))
model.add(ELU())
model.add(pooling.MaxPooling2D(pool_size=(2, 2)))
model.add(convolutional.Convolution2D(64, 3, 3))
model.add(pooling.MaxPooling2D(pool_size=(2, 2)))
model.add(core.Flatten())
model.add(core.Dense(500))
model.add(ELU())
model.add(core.Dropout(.5))
model.add(core.Dense(100))
model.add(ELU())
model.add(core.Dropout(.25))
model.add(core.Dense(20))
model.add(ELU())
model.add(core.Dense(1))
model.compile(optimizer=optimizers.Adam(lr=1e-04), loss='mean_squared_error')
model.fit_generator(generator(train_data, augment=True),
samples_per_epoch=train_data.shape[0],
					nb_epoch=20,
					validation_data=generator(valid_data, augment=False),
					nb_val_samples=valid_data.shape[0],
					callbacks=[weight_logger()], # Add a callbacks to save best model in each epoch, but it is not neccesary
					verbose=1)  # If verbose=1 or none, will show processbar, keep it if run without GPU

### Save Model, don't use model.save_weight in here, it save weight only, autonomous vehicle will not run
model.save('model.h5')
