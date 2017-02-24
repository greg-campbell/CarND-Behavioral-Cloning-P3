import numpy as np
import pandas as pd
import cv2

import matplotlib.pyplot as plt
from IPython.display import display as display
from IPython.core.pylabtools import figsize, getfigs
#import math
#import matplotlib.image as mpimg
#import matplotlib.transforms as mtransforms
def draw(img, title = '', filename='', color = cv2.COLOR_BGR2RGB):
    color_img = cv2.cvtColor(img, color)
    """ Draw a single image with a title """
    f = plt.figure()
    plt.title(title)
    plt.imshow(color_img, cmap='gray')
    #display(f)
    if filename != '':
        plt.savefig('./img/'+filename+'.png')
    plt.close(f)

def read_row(row):
  img = read_image(row['center'][0].strip())
  angle = row['steering'][0]
  #if 1 == np.random.randint(2):
  #    img = cv2.flip(img, 1)
  #    angle = -angle
  return preprocess(img), angle

def augment_row(row):
  cameras = ['left', 'center', 'right']
  offsets = [1, 0, -1]
  camera_index = np.random.randint(3)
  img = read_image(row[cameras[camera_index]][0].strip())
  angle = row['steering'][0] + offsets[camera_index]*0.28
  
  if 1 == np.random.randint(2):
      img = cv2.flip(img, 1)
      angle = -angle
  return preprocess(img), angle

def read_image(path):
    return cv2.imread('./data/' + path)

row_angles = []
from numpy.random import choice
def prepare_row_angles(rows):
    for angle in rows:
        row_angles.extend([angle])
    hist, bin_edges = np.histogram(row_angles)
    hist_vals = [hist[np.searchsorted(bin_edges, row_angles[i])-1] for i in range(len(row_angles))]
    ps = [1/x for x in hist_vals]
    #ps = [hist_vals[i] * (abs(row_angles[i]) + 1.0) for i in range(len(row_angles))]
    pps = [x/sum(ps) for x in ps]
    return pps

def uniform_select_row_num(n):
   draw = choice([i for i in range(len(row_angles))], n)#, p=pps)
   return draw

# Read in the driving log CSV
driving_log = pd.read_csv('./data/driving_log.csv')
driving_log.head()

test_image = cv2.imread('./data/' + driving_log['center'][0])

#draw(test_image, "Original image (center camera)")

height, width, channels = test_image.shape
print("Image dimensions:" , width, height)

def crop_image(img, y1, y2):
   #return img[y1:y2,:] 
   return img

def resize_image(img, width, height):
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

def preprocess(img):
    return resize_image(crop_image(img, 53, 128), 64, 32 )

preprocessed = preprocess(test_image)
print(preprocessed.shape)
#draw(preprocessed, "After preprocessing (crop + resize)", "preprocessed")

from keras.models import Sequential
from keras.layers.core import Activation, Dense, Dropout, Flatten
#from keras.activations import relu
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers import Cropping2D, Lambda

model = Sequential()
model.add(Lambda(lambda x: (x / 127.5) - 1.0, input_shape=preprocessed.shape))
model.add(Cropping2D(cropping=((10, 4), (0, 0)), input_shape=preprocessed.shape))
model.add(Convolution2D(64, 2, 2, border_mode='same', activation='relu'))#,input_shape=(16,32,3)))
model.add(MaxPooling2D((2, 2)))
model.add(Convolution2D(32, 3, 3, border_mode='same',activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dropout(0.3))
model.add(Dense(1024, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1))

from sklearn.model_selection import train_test_split
data_train, data_val = train_test_split(driving_log)

def generate_batch_from_data(data, filter_probability, size = 1024):
    images = np.zeros((size, *preprocessed.shape))
    steering_angles = np.zeros(size)
    row_indices = uniform_select_row_num(size)
    while 1:
        for i in range(size):
            row = data.iloc[[row_indices[i]]].reset_index()
            #if np.random.uniform() < filter_probability:
            #  while abs(row['steering'][0]) > 0.1: 
            #    row = data.iloc[[np.random.randint(len(data))]].reset_index()
            augment_var = np.random.uniform() < 0.3
            if augment_var:
              x,y = augment_row(row)
            else:
              x,y = read_row(row)
            images[i] = x
            steering_angles[i] = y
        yield images, steering_angles

def generate_batch_for_val(data, size = 1024):
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
#for epoch in range(5):
#    filter_probability = 1.0/(1+epoch)
#    history = model.fit_generator(generate_batch_from_data(data_train, filter_probability),
#                                  samples_per_epoch=4000,
#                                  validation_data = generate_batch_for_val(data_val),
#                                  nb_val_samples=400,
#                                  nb_epoch=1, verbose=2)

pps = prepare_row_angles(data_train['steering'])
history = model.fit_generator(generate_batch_from_data(data_train, 0.0),
                              #samples_per_epoch=32*len(data_train)//32, nb_epoch=10,
                              samples_per_epoch=len(data_train)*10, nb_epoch=10,
                              validation_data = generate_batch_for_val(data_val),
                              nb_val_samples = len(data_val),
                              #nb_val_samples = 400,
                              verbose = 2)

model.save('model.h5')
