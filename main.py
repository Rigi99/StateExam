import fnmatch
import os
from glob import glob
from random import seed
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import layers, Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.saving.save import load_model
from keras.utils import to_categorical
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split


def if_pneumonia(x):
    return 'PNEUMONIA' in x


def if_normal(x):
    return 'NORMAL' in x


print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

imagePaths = glob('C:/Users/user/OneDrive/Pictures/chest_xray/**/**/*.jpeg', recursive=False)

patternNormal = '*NORMAL*'
patternBacteria = '*_bacteria_*'
patternVirus = '*_virus_*'

normal = fnmatch.filter(imagePaths, patternNormal)
bacteria = fnmatch.filter(imagePaths, patternBacteria)
virus = fnmatch.filter(imagePaths, patternVirus)

x = []
y = []

counter = 1

if os.path.exists("x.npy") and os.path.exists("y.npy"):
    x = np.load('x.npy')
    y = np.load('y.npy')
else:
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

xTrain, xValid, yTrain, yValid = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
yTrain = to_categorical(yTrain, num_classes=2)
yValid = to_categorical(yValid, num_classes=2)
del x, y, normal, virus, bacteria, counter

fig, m_axs = plt.subplots(2, 4, figsize=(20, 10))
for (c_x, c_y, c_ax) in zip(xValid, yValid, m_axs.flatten()):
    c_ax.imshow(c_x[:, :, 0], cmap='bone')
    if c_y[0] == 1:
        c_ax.set_title('Normal')
    else:
        c_ax.set_title('Pneumonia')
    c_ax.axis('off')
plt.show()

np.random.seed(111)
seed(111)
chanelAxis = -1


def network():
    model = Sequential([
        layers.Input(shape=(224, 224, 3)),

        layers.Conv2D(32, 3, padding='same', use_bias=False),
        layers.BatchNormalization(axis=-1),
        layers.Activation('relu'),
        layers.Conv2D(32, 3, padding='same', use_bias=False),
        layers.BatchNormalization(axis=-1),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'),

        layers.Conv2D(64, 3, padding='same', use_bias=False),
        layers.BatchNormalization(axis=-1),
        layers.Activation('relu'),
        layers.Conv2D(64, 3, padding='same', use_bias=False),
        layers.BatchNormalization(axis=-1),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'),

        layers.Conv2D(128, 3, padding='same', use_bias=False),
        layers.BatchNormalization(axis=-1),
        layers.Activation('relu'),
        layers.Conv2D(128, 3, padding='same', use_bias=False),
        layers.BatchNormalization(axis=-1),
        layers.Activation('relu'),
        layers.MaxPooling2D((3, 3), strides=(3, 3), padding='same'),

        layers.Conv2D(256, 3, padding='same', use_bias=False),
        layers.BatchNormalization(axis=-1),
        layers.Activation('relu'),
        layers.Conv2D(256, 3, padding='same', use_bias=False),
        layers.BatchNormalization(axis=-1),
        layers.Activation('relu'),
        layers.MaxPooling2D((3, 3), strides=(3, 3), padding='same'),

        layers.Conv2D(1024, 3, padding='same', use_bias=False),
        layers.BatchNormalization(axis=-1),
        layers.Activation('relu'),
        layers.Conv2D(512, 3, padding='same', use_bias=False),
        layers.Dropout(0.4),
        layers.BatchNormalization(axis=-1),
        layers.Activation('relu'),
        layers.MaxPooling2D((3, 3), strides=(3, 3), padding='same'),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dense(1024, activation='relu'),
        layers.Dense(512, activation='relu'),
        layers.Dense(512, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(2, activation='softmax')
    ])
    return model


myCnn = network()
myCnn.summary()

LEARN_RATE = 1e-4

myCnn.compile(optimizer=Adam(learning_rate=LEARN_RATE), loss='categorical_crossentropy',
              metrics=['categorical_accuracy'])

checkpoint = ModelCheckpoint(filepath='model.h5', save_best_only=True, monitor='val_loss', mode='min')

reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=5, verbose=1, mode='auto', min_delta=0.0001,
                                   cooldown=5, min_lr=0.0001)

earlyStopping = EarlyStopping(monitor='val_loss', patience=50, mode='min')

callbacksList = [checkpoint, earlyStopping, reduceLROnPlat]

# myCnn.load_weights('model.h5')

history = myCnn.fit(xTrain, yTrain, batch_size=16,
                    epochs=10, verbose=1, validation_split=0.2, callbacks=callbacksList)

testLoss, testScore = myCnn.evaluate(xValid, yValid, batch_size=16)
print("Loss on test set: ", testLoss)
print("Accuracy on test set: ", testScore)

yPred = myCnn.predict(xValid, callbacks=callbacksList)

# Original labels
originalTestLabels = np.argmax(yValid, axis=-1)

print(originalTestLabels.shape)
print(yPred.shape)

print(classification_report(np.argmax(yValid, axis=1), np.argmax(yPred, axis=1)))

# Get the confusion matrix
cm = confusion_matrix(np.argmax(yValid, axis=1), np.argmax(yPred, axis=1))
plt.figure()
plot_confusion_matrix(cm, figsize=(12, 8), hide_ticks=True, cmap=plt.cm.Blues)
plt.xticks(range(2), ['Normal', 'Pneumonia'], fontsize=16)
plt.yticks(range(2), ['Normal', 'Pneumonia'], fontsize=16)
plt.show()

# Calculate Precision and Recall
tn, fp, fn, tp = cm.ravel()
precision = tp / (tp + fp)
recall = tp / (tp + fn)

print("Recall of the model is {:.2f}".format(recall))
print("Precision of the model is {:.2f}".format(precision))

fpr, tpr, _ = roc_curve(np.argmax(yValid, -1) == 0, yPred[:, 0])
fig, ax1 = plt.subplots(1, 1, figsize=(5, 5), dpi=250)
ax1.plot(fpr, tpr, 'b.-',
         label='Own-Model (AUC:%2.2f)' % roc_auc_score(np.argmax(yValid, -1) == 0, yPred[:, 0]))
ax1.plot(fpr, fpr, 'k-', label='Random Guessing')
ax1.legend(loc=4)
ax1.set_xlabel('False Positive Rate')
ax1.set_ylabel('True Positive Rate')
ax1.set_title('Pneumonia Classification ROC Curve')
fig.savefig('roc_valid.pdf')
plt.show()

# Save model
myCnn.save(os.path.join('models', 'pneumoniaModel.h5'))

# myCnn = load_model(os.path.join('models', 'pneumoniaModel.h5'))

normalImagePaths = glob('C:/Users/user/OneDrive/Pictures/chest_xray/test/NORMAL/*.jpeg', recursive=False)
pneumoniaImagePaths = glob('C:/Users/user/OneDrive/Pictures/chest_xray/test/PNEUMONIA/*.jpeg', recursive=False)
predictNormal = []
predictPneumonia = []
for i, j in zip(normalImagePaths[:230], pneumoniaImagePaths[:230]):
    img = tf.keras.utils.load_img(
        i, target_size=(224, 224)
    )

    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    predictions = myCnn.predict(img_array)
    if np.argmax(predictions) == 0:
        predictNormal.append('NORMAL')
    else:
        predictNormal.append('PNEUMONIA')

    img = tf.keras.utils.load_img(
        j, target_size=(224, 224)
    )

    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    predictions = myCnn.predict(img_array)
    if np.argmax(predictions) == 1:
        predictPneumonia.append('PNEUMONIA')
    else:
        predictPneumonia.append('NORMAL')

print(predictPneumonia)
print(len(predictPneumonia))
print(sum(if_pneumonia(i) for i in predictPneumonia) * 100 / len(predictPneumonia))
print('**************')
print(predictNormal)
print(len(predictNormal))
print(sum(if_normal(i) for i in predictNormal) * 100 / len(predictNormal))
