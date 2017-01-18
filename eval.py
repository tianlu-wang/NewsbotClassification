from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
import numpy as np

import resnet

nb_classes = 12
img_rows, img_cols = 224, 224
img_channels = 3
data_dir = "littletest"

model = resnet.ResnetBuilder.build_resnet_50((img_rows, img_cols, img_channels), nb_classes)
model.load_weights("%s/weights.best.hdf5" % data_dir)
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
X_test = np.load("%s/x_test.data" % data_dir)
y_test = np.load("%s/y_test.data" % data_dir)
Y_test = np_utils.to_categorical(y_test, nb_classes)
X_test = X_test.astype('float32')
X_test /= 255
scores = model.evaluate(X_test, Y_test, verbose=1)
print("%s: %.4f%%" % (model.metrics_names[1], scores[1]*100))

predictions = model.predict(X_test, verbose=1)
test_pred_file = open("%s/test.pred" % data_dir, "wb")
np.save(test_pred_file, predictions)