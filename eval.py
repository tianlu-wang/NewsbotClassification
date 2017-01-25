from __future__ import print_function
import sys, os
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
import numpy as np
from keras.models import Model
from keras.optimizers import SGD
from keras.layers import Input, Activation, merge, Dense, Flatten
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16

import resnet
def eval(test_data_dir, weights_name, model_name, pretrained, optimizer_name):
	nb_classes = 12
	img_rows, img_cols = 224, 224
	img_channels = 3
	data_dir = test_data_dir

	if model_name == "my_resnet":
		model = resnet.ResnetBuilder.build_resnet_50((img_rows, img_cols, img_channels), nb_classes)
	elif model_name == "keras_resnet":
		if pretrained:
			base_model = ResNet50(include_top=False, weights="imagenet", input_tensor=None, input_shape=(224, 224, 3))
			for layer in base_model.layers:
			    layer.trainable = False
		else:
			base_model = ResNet50(include_top=False, weights=None, input_tensor=None, input_shape=(224, 224, 3))

		x = base_model.output
		x = Flatten(name="flatten")(x)
		x = Dense(nb_classes, activation='softmax', name='fc12')(x)
		model = Model(input=base_model.input, output=x)
	elif model_name == "VGG16":
		if pretrained:
			base_model = VGG16(include_top=False, weights="imagenet", input_tensor=None, input_shape=(224, 224, 3))
			for layer in base_model.layers:
			    layer.trainable = False
		else:
			base_model = VGG16(include_top=False, weights=None, input_tensor=None, input_shape=(224, 224, 3))

		x = base_model.output
		x = Flatten(name="flatten")(x)
		x = Dense(4096, activation='relu', name='fc1')(x)
		x = Dense(4096, activation='relu', name='fc2')(x)
		x = Dense(nb_classes, activation='softmax', name='predictions')(x)
		model = Model(input=base_model.input, output=x)

	model.load_weights("%s/%s" % (data_dir, weights_name))
	if optimizer_name == 'sgd':
		sgd = SGD(lr=0.0005, decay=1e-6, momentum=0.9, nesterov=True)
		optimizer = sgd
	elif optimizer_name == 'ramsprop':
		pass
	elif optimizer_name == 'adagrad':
		pass
	elif optimizer_name == 'adadelta':
		pass
	elif optimizer_name == 'adam':
		pass
	elif optimizer_name == 'adamax':
		pass
	elif optimizer_name == 'nadam':
		pass
	
	model.compile(loss='categorical_crossentropy',
	              optimizer=optimizer_name,
	              metrics=['accuracy'])
	X_test = np.load("%s/x_test.data" % data_dir)
	y_test = np.load("%s/y_test.data" % data_dir)
	Y_test = np_utils.to_categorical(y_test, nb_classes)
	X_test = X_test.astype('float32')
	X_test /= 255
	scores = model.evaluate(X_test, Y_test, verbose=1)
	print("%s: %.4f%%" % (model.metrics_names[1], scores[1]*100))

	predictions = model.predict(X_test, verbose=1)
	test_pred_file = open("%s/%s.pred" % (data_dir, weights_name), "wb")
	np.save(test_pred_file, predictions)

if __name__ == '__main__':
	if len(sys.argv) != 6:
		print('USAGE: python eval.py <test_data_dir> <weights_name> <model_name> <pretrained> <optimizer_name>')
	else:
		test_data_dir = sys.argv[1]
		weights_name = sys.argv[2]
		model_name = sys.argv[3]
		pretrained = sys.argv[4]
		optimizer_name = sys.argv[5]
		eval(test_data_dir, weights_name, model_name, int(pretrained), optimizer_name)