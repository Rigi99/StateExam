import fnmatch
import os
from glob import glob
from random import seed
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.utils import to_categorical
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
import keras.regularizers
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, AveragePooling2D, Concatenate, \
    BatchNormalization, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from glob import glob
from sklearn.metrics import confusion_matrix
from tensorflow.keras import regularizers


def if_pneumonia(x):
    return 'PNEUMONIA' in x


def if_normal(x):
    return 'NORMAL' in x


img_input = Input(shape=(224, 224, 3))


def inception_module(x, filters):
    path1 = Conv2D(filters=filters[0], kernel_size=1, activation='relu')(x)

    path2 = Conv2D(filters=filters[1], kernel_size=1, activation='relu')(x)
    path2 = Conv2D(filters=filters[2], kernel_size=3, padding='same', activation='relu')(path2)

    path3 = Conv2D(filters=filters[3], kernel_size=1, activation='relu')(x)
    path3 = Conv2D(filters=filters[4], kernel_size=5, padding='same', activation='relu')(path3)

    path4 = MaxPooling2D(pool_size=3, strides=1, padding='same')(x)
    path4 = Conv2D(filters=filters[5], kernel_size=1, activation='relu')(path4)

    return Concatenate(axis=-1)([path1, path2, path3, path4])


def googlenet_model(img_input):
    x = Conv2D(filters=64, kernel_size=7, strides=2, padding='same', activation='relu')(img_input)
    x = MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

    x = Conv2D(filters=64, kernel_size=1, strides=1, padding='valid', activation='relu')(x)
    x = Conv2D(filters=192, kernel_size=3, strides=1, padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

    x = inception_module(x, filters=[64, 96, 128, 16, 32, 32])
    x = MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

    x = inception_module(x, filters=[128, 128, 192, 32, 96, 64])
    x = inception_module(x, filters=[192, 96, 208, 16, 48, 64])
    x = MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

    x = inception_module(x, filters=[160, 112, 224, 24, 64, 64])
    x = inception_module(x, filters=[128, 128, 256, 24, 64, 64])
    x = inception_module(x, filters=[112, 144, 288, 32, 64, 64])
    x = inception_module(x, filters=[256, 160, 320, 32, 128, 128])
    x = inception_module(x, filters=[256, 160, 320, 32, 128, 128])
    x = inception_module(x, filters=[384, 192, 384, 48, 128, 128])
    x = AveragePooling2D(pool_size=7, strides=1, padding='valid')(x)
    x = Dropout(rate=0.4)(x)

    x = Flatten()(x)
    x = Dense(units=512, activation='relu')(x)
    x = Dense(units=256, activation='relu')(x)
    x = Dense(units=64, activation='relu', kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
              bias_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
              activity_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4))(x)
    x = Dropout(rate=0.4)(x)
    x = Dense(units=2, activation='softmax')(x)

    model = Model(inputs=img_input, outputs=x)

    return model


model = googlenet_model(img_input)
model.summary()

LEARN_RATE = 1e-5

model.compile(optimizer=Adam(learning_rate=LEARN_RATE), loss='categorical_crossentropy',
              metrics=['categorical_accuracy'])

weight_path = "{}.best_only.hdf5".format('save')

checkpoint = ModelCheckpoint(weight_path, monitor='loss', verbose=1,
                             save_best_only=True, mode='min', save_weights_only=True)

reduceLROnPlat = ReduceLROnPlateau(monitor='loss', factor=0.8, patience=5, verbose=1, mode='auto', min_delta=0.0001,
                                   cooldown=5, min_lr=0.0001)

earlyStopping = EarlyStopping(monitor='loss', patience=50, mode='min')

callbacksList = [checkpoint, earlyStopping, reduceLROnPlat]

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

imagePaths = glob('/home/diak2021/RigmanyiZsombor/chest_xray/train/**/*.jpeg', recursive=False)
imagePaths += glob('/home/diak2021/RigmanyiZsombor/chest_xray/test/**/*.jpeg', recursive=False)

patternNormal = '*NORMAL*'
patternBacteria = '*_bacteria_*'
patternVirus = '*_virus_*'

normal = fnmatch.filter(imagePaths, patternNormal)
bacteria = fnmatch.filter(imagePaths, patternBacteria)
virus = fnmatch.filter(imagePaths, patternVirus)

x = []
y = []

counter = 1

for img in imagePaths:
    counter += 1
    if counter % 100 == 0:
        print('.', end='', flush=True)
    fullSizeImage = cv2.imread(img)
    im = cv2.resize(fullSizeImage, (224, 224), interpolation=cv2.INTER_CUBIC)
    del fullSizeImage
    im = im.astype(np.float32) / 255.
    x.append(im)
    if img in normal:
        y.append(0)
    elif img in bacteria:
        y.append(1)
    elif img in virus:
        y.append(1)
    else:
        print('no class')
print()

os.environ['PYTHONHASHSEED'] = '0'

x = np.array(x)
y = np.array(y)

np.save('x.npy', x)
np.save('y.npy', y)

xTrain, xValid, yTrain, yValid = train_test_split(x, y, test_size=0.3, random_state=101)

yTrain = to_categorical(yTrain, num_classes=2)
yValid = to_categorical(yValid, num_classes=2)

history = model.fit(xTrain, yTrain, epochs=75, batch_size=32, verbose=1, callbacks=callbacksList)

pred_y = model.predict(xValid)

orig_test_labels = np.argmax(yValid, axis=-1)

cm = confusion_matrix(np.argmax(yValid, axis=1), np.argmax(pred_y, axis=1))
plt.figure()
plot_confusion_matrix(cm, figsize=(12, 8), hide_ticks=True, cmap=plt.cm.Blues)
plt.xticks(range(2), ['Normal', 'Pneumonia'], fontsize=16)
plt.yticks(range(2), ['Normal', 'Pneumonia'], fontsize=16)
plt.show()

# Calculate Precision and Recall
tp, fp, fn, tn = cm.ravel()
print("tp", tp)
print("fp", fp)
print("fn", fn)
print("tn", tn)
precision = tp / (tp + fp)
recall = tp / (tp + fn)
accuracy = (tp + tn) / (tp + tn + fn + fp)
dice_score = (2 * tp) / ((2 * tp) + fp + fn)

print("Recall of the model is {:.2f}".format(recall))
print("Precision of the model is {:.2f}".format(precision))
print("Accuracy of the model is {:.2f}".format(accuracy))
print("Dice score of the model is {:.2f}".format(dice_score))

normalImagePaths = glob('/home/diak2021/RigmanyiZsombor/chest_xray/val/NORMAL/*.jpeg', recursive=False)
pneumoniaImagePaths = glob('/home/diak2021/RigmanyiZsombor/chest_xray/val/PNEUMONIA/*.jpeg', recursive=False)
predictNormal = []
predictPneumonia = []

for i, j in zip(normalImagePaths, pneumoniaImagePaths):
    fullSizeImage = cv2.imread(i)
    im = cv2.resize(fullSizeImage, (224, 224), interpolation=cv2.INTER_CUBIC)
    im = im.astype(np.float32) / 255.
    im = tf.expand_dims(im, 0)
    predictions = model.predict(im)
    if np.argmax(predictions) == 0:
        predictNormal.append('NORMAL')
    else:
        predictNormal.append('PNEUMONIA')

    fullSizeImage = cv2.imread(j)
    im = cv2.resize(fullSizeImage, (224, 224), interpolation=cv2.INTER_CUBIC)
    im = im.astype(np.float32) / 255.
    im = tf.expand_dims(im, 0)
    predictions = model.predict(im)
    if np.argmax(predictions) == 1:
        predictPneumonia.append('PNEUMONIA')
    else:
        predictPneumonia.append('NORMAL')

print('PNEUMONIA')
print(sum(if_pneumonia(i) for i in predictPneumonia) * 100 / len(predictPneumonia))
print('**************')
print('NORMAL')
print(sum(if_normal(i) for i in predictNormal) * 100 / len(predictNormal))

model.save(os.path.join('models', 'pneumoniaModel.h5'))
