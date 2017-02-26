import numpy as np
import pandas as pd
import cv2

import matplotlib.pyplot as plt
from IPython.display import display as display
from IPython.core.pylabtools import figsize, getfigs

""" Draw a single image with a title 
    If filename is specified, the image is saved to ./img/$filename.png
"""
def draw(img, title = '', filename='', color = cv2.COLOR_BGR2RGB):
    color_img = cv2.cvtColor(img, color)
    f = plt.figure()
    plt.title(title)
    plt.imshow(color_img, cmap='gray')
    if filename != '':
        plt.savefig('./img/'+filename+'.png')
    else:
        display(f)
    plt.close(f)

""" Read and preprocess the center camera and steering angle of a single row of data without augmentation. """
def read_row(row):
  img = read_image(row['center'][0].strip())
  angle = row['steering'][0]
  return preprocess(img), angle

""" Read and preprocess the camera and steering angle of a single row of data with augmentation. """
def augment_row(row):
  cameras = ['left', 'center', 'right']
  offsets = [1, 0, -1]

  # Randomly select a camera.
  camera_index = np.random.randint(3)
  img = read_image(row[cameras[camera_index]][0].strip())

  # Adjust angle by +/- 0.28 for left or right cameras, respectively
  angle = row['steering'][0] + offsets[camera_index]*0.28
  
  # Flip the image horizontally and reverse the angle with 50% probability
  if 1 == np.random.randint(2):
      img = cv2.flip(img, 1)
      angle = -angle
  return preprocess(img), angle

""" Read an image from the data directory. """
def read_image(path):
    return cv2.imread('./data/' + path)

row_angles = []
from numpy.random import choice

""" Prepare row angles in the training set for uniform selection.
    The data set is binned into 10 bins of width $$ 1/10 * (max(angle) - min(angle)) $$
    and a probability for each row is calculated such that a random selection will have
    an equal probability of selecting a row from each bin.
"""
def prepare_row_angles(rows):
    for angle in rows:
        row_angles.extend([angle])
    hist, bin_edges = np.histogram(row_angles)
    hist_vals = [hist[np.searchsorted(bin_edges, row_angles[i])-1] for i in range(len(row_angles))]
    ps = [1/x for x in hist_vals]
    pps = [x/sum(ps) for x in ps]
    return pps

""" Select $n row indices from the training set such that there is an equal chance
    of selecting a given row from each bin.
""" 
def uniform_select_row_num(n):
   draw = choice([i for i in range(len(row_angles))], n, p=pps)
   return draw

# Read in the driving log CSV
driving_log = pd.read_csv('./data/driving_log.csv')
driving_log.head()

test_image = cv2.imread('./data/' + driving_log['center'][0])
draw(test_image, "Original image (center camera)", "original")

height, width, channels = test_image.shape
print("Image dimensions:" , width, height)

""" Crop the height of the image from $y1 to $y2. """
def crop_image(img, y1, y2):
   return img[y1:y2,:] 

""" Resize img to $width x $height. """
def resize_image(img, width, height):
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

""" Preprocess image by resizing it to 64x32. """
def preprocess(img):
    return resize_image(img, 64, 32)

preprocessed = preprocess(test_image)
print(preprocessed.shape)
draw(preprocessed, "After preprocessing (resize only)", "preprocessed")
draw(crop_image(preprocessed, 10, 18), "After preprocessing (resize and crop)", "preprocessed2")

from keras.models import Sequential
from keras.layers.core import Activation, Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers import Cropping2D, Lambda

model = Sequential()
model.add(Lambda(lambda x: (x / 127.5) - 1.0, input_shape=preprocessed.shape))
model.add(Cropping2D(cropping=((10, 4), (0, 0)), input_shape=preprocessed.shape))
model.add(Convolution2D(64, 2, 2, border_mode='same', activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Convolution2D(32, 3, 3, border_mode='same',activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dropout(0.3))
model.add(Dense(1024, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1))

from sklearn.model_selection import train_test_split
data_train, data_val = train_test_split(driving_log)

print("Training set:" , len(data_train))
print("Validation set:" , len(data_val))

""" Generator for training. """
def generate_batch_from_data(data, size = 1024):
    images = np.zeros((size, *preprocessed.shape))
    steering_angles = np.zeros(size)
    row_indices = uniform_select_row_num(size)
    while 1:
        for i in range(size):
            row = data.iloc[[row_indices[i]]].reset_index()
            augment_var = np.random.uniform() < 0.3
            if augment_var:
              x,y = augment_row(row)
            else:
              x,y = read_row(row)
            images[i] = x
            steering_angles[i] = y
        yield images, steering_angles

""" Generator for validation. """
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

# Prepare row angle probabilities
pps = prepare_row_angles(data_train['steering'])


from keras.utils.visualize_util import plot
plot(model, to_file='./img/model.png')

history = model.fit_generator(generate_batch_from_data(data_train),
                              samples_per_epoch=len(data_train)*10, nb_epoch=10,
                              validation_data = generate_batch_for_val(data_val),
                              nb_val_samples = len(data_val),
                              verbose = 2)


def plot_hist_loss(history):
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    plt.savefig('./img/accuracy.png')
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    plt.savefig('./img/loss.png')

plot_hist_loss(history)

model.save('model_x.h5')
