from keras import models, layers
from keras.callbacks import ModelCheckpoint

import numpy as np
import os
import cv2

np.random.seed(1)

train_images = []
train_labels = []
test_images = []
test_labels = []

for folder in os.listdir('idenprof/test'):
    if folder != '.DS_Store':
        for file in os.listdir('idenprof/test/' + folder):
            if file.endswith('.jpg'):
                img = cv2.imread(os.path.join('idenprof/test/',folder,file))
                image_resized = cv2.resize(img, (128, 128))
                test_images.append(image_resized)
                test_labels.append(folder)

for folder in os.listdir('idenprof/train'):
    if folder != '.DS_Store':
        for file in os.listdir('idenprof/train/' + folder):
            if file.endswith('.jpg'):
                img = cv2.imread(os.path.join('idenprof/train',folder,file))
                image_resized = cv2.resize(img, (128, 128))
                train_images.append(image_resized)
                train_labels.append(folder)

train_images = np.array(train_images)/255.0
test_images = np.array(test_images)/255.0

#converting the y_data into categorical:
from sklearn.preprocessing import LabelEncoder
train_labels_encoded = LabelEncoder().fit_transform(train_labels)
test_labels_encoded = LabelEncoder().fit_transform(test_labels)
from keras.utils import to_categorical
train_labels_categorical = to_categorical(train_labels_encoded)
test_labels_categorical = to_categorical(test_labels_encoded)

#lets shuffle all the data we have:
r = np.arange(train_images.shape[0])
np.random.seed(42)
np.random.shuffle(r)
train_images = train_images[r]
train_labels = train_labels_categorical[r]
test_labels = test_labels_categorical

#X_train, X_val, y_train, y_val = train_test_split(train_images, train_labels, test_size=0.2, random_state=2)

model = models.Sequential()
model.add(layers.Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=train_images.shape[1:]))
model.add(layers.MaxPool2D(pool_size=(2, 2)))
model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(layers.MaxPool2D(pool_size=(2, 2)))
model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
model.add(layers.MaxPool2D(pool_size=(2, 2)))
model.add(layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu'))
model.add(layers.MaxPool2D(pool_size=(2, 2)))
model.add(layers.Dropout(rate=0.25))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(rate=0.5))
model.add(layers.Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# model.summary()

checkpoint = ModelCheckpoint("model_weights.h5", monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]


history = model.fit(train_images, train_labels, epochs=20, batch_size=100, callbacks=callbacks_list, validation_split=0.3)

test_pred = model.predict_classes(test_images)

#converting over Y test to actual labels.
test_labels = np.argmax(test_labels, axis = 1)
from sklearn.metrics import accuracy_score
print('the accuracy obtained on the test set is:', accuracy_score(test_pred,test_labels))

import matplotlib.pyplot as plt

plt.figure(figsize = (8,8))
plt.plot(history.history['accuracy'], label='training accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title('Loss/accuracy')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()
