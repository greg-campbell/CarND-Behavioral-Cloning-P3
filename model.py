import numpy as np
import pandas as pd
import cv2

### Data exploration visualization goes here.
### Feel free to use as many code cells as needed.
import matplotlib.pyplot as plt
from IPython.display import display as display
#from IPython.core.pylabtools import figsize, getfigs
#import math
#import matplotlib.image as mpimg
#import matplotlib.transforms as mtransforms
def draw(img, title = '', color = cv2.COLOR_BGR2RGB):
    color_img = cv2.cvtColor(img, color)
    """ Draw a single image with a title """
    f = plt.figure()
    plt.title(title)
    plt.imshow(color_img, cmap='gray')
    display(f)
    plt.close(f)

def read_row(row):
  img = read_image(row['center'][0])
  angle = row['steering'][0]
  return preprocess(img), angle
  

def read_image(path):
    return cv2.imread('./data/' + path)

# Read in the driving log CSV
driving_log = pd.read_csv('./data/driving_log.csv')
driving_log.head()

test_image = cv2.imread('./data/' + driving_log['center'][0])

#draw(test_image, "Original image (center camera)")

height, width, channels = test_image.shape
print("Image dimensions:" , width, height)

def crop_image(img, x, y, w, h):
   return img[y:y+h,:] 
#def crop_image(img, x, y, w, h):
#    y2 = y+h
#    x2 = x+w
#    return img[y:y2, x:x2]
#
def resize_image(img, width, height):
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

def preprocess(img):
    return resize_image(crop_image(img, 0, 40, 320, 40), 200, 66 )
#
preprocessed = preprocess(test_image)
print(preprocessed.shape)
#draw(preprocessed, "After preprocessing (crop + resize)")

from keras.models import Sequential
from keras.layers.core import Activation, Dense, Dropout, Flatten
#from keras.activations import relu
from keras.layers.convolutional import Convolution2D
from keras.layers import Cropping2D, Lambda

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=preprocessed.shape))
#model.add(Cropping2D(cropping=((20, 10), (0, 0)), input_shape=preprocessed.shape))
model.add(Convolution2D(24, 5, 5, subsample=(2,2)))
model.add(Activation('relu'))
model.add(Convolution2D(36, 5, 5, subsample=(2,2)))
model.add(Activation('relu'))
model.add(Convolution2D(48, 5, 5, subsample=(2,2)))
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(1024))
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(1))

from sklearn.model_selection import train_test_split
data_train, data_val = train_test_split(driving_log)

def generate_batch_from_data(data, size = 32):
    images = np.zeros((size, *preprocessed.shape))
    steering_angles = np.zeros(size)
    while 1:
        for i in range(size):
            row = data.iloc[[np.random.randint(len(data))]].reset_index()
            x,y = read_row(row)
            images[i] = x
            steering_angles[i] = y
        yield images, steering_angles


model.compile('adam', 'mean_squared_error', ['mean_squared_error'])

history = model.fit_generator(generate_batch_from_data(data_train),
                              samples_per_epoch=32*len(data_train)//32, nb_epoch=5,
                              validation_data = generate_batch_from_data(data_val),
                              nb_val_samples = len(data_val),
                              verbose = 2)

#with open('model.json', 'w') as json:
#    json.write(model.to_json())
#model.save_weights('model.h5')
model.save('model.h5')
