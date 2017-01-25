from __future__ import print_function
import sys
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD
from keras.layers import Input, Activation, merge, Dense, Flatten
import numpy as np
from keras.models import Model
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
import resnet
from utils import PerClassMetric

def train(data_dir, model_name, pretrained, optimizer_name, weight_path):
	# parameters
	batch_size = 32
	nb_classes = 12
	nb_epoch = 50
	data_augmentation = False

	# input image dimensions and channels
	img_rows, img_cols = 224, 224
	img_channels = 3
	# data_dir = "10000test"

	# read data
	X_train = np.load("%s/x_train.data" % data_dir)
	y_train = np.load("%s/y_train.data" % data_dir)
	X_valid = np.load("%s/x_valid.data" % data_dir)
	y_valid = np.load("%s/y_valid.data" % data_dir)

	print('X_train shape:', X_train.shape)
	print(X_train.shape[0], 'train samples')
	print(X_valid.shape[0], 'valid samples')

	# Convert class vectors to binary class matrices.(one hot coder)
	Y_train = np_utils.to_categorical(y_train, nb_classes)
	Y_valid = np_utils.to_categorical(y_valid, nb_classes)

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
		# model.load_weights("finetune/data_0.45/weights_keras_resnet1sgd.best.hdf5")
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

	if weight_path == "no_weights":
		pass
	else:
		model.load_weights(weight_path)

	# adjust parameters
	if optimizer_name == 'sgd':
		sgd = SGD(lr=0.0005, decay=1e-6, momentum=0.9, nesterov=True)
		optimizer = sgd
	elif optimizer_name == 'ramsprop':
		optimizer = optimizer_name
	elif optimizer_name == 'adagrad':
		optimizer = optimizer_name
	elif optimizer_name == 'adadelta':
		optimizer = optimizer_name
	elif optimizer_name == 'adam':
		optimizer = optimizer_name
	elif optimizer_name == 'adamax':
		optimizer = optimizer_name
	elif optimizer_name == 'nadam':
		optimizer = optimizer_name
	
	model.compile(loss='categorical_crossentropy',
	              optimizer=optimizer,
	              metrics=['accuracy'])

	X_train = X_train.astype('float32')
	X_valid = X_valid.astype('float32')
	X_train /= 255
	X_valid /= 255

	checkpoint = ModelCheckpoint("%s/weights_%s.best.hdf5" % (data_dir, model_name+str(pretrained)+optimizer_name), monitor='val_acc',
	                             verbose=1, save_best_only=True, mode='max')
	per_class_metric = PerClassMetric(X_valid, Y_valid, batch_size)
	callbacks_list = [checkpoint, per_class_metric]

	if not data_augmentation:
		print('Not using data augmentation.')
		model.fit(X_train, Y_train,
			batch_size=batch_size,
			nb_epoch=nb_epoch,
			validation_data=(X_valid, Y_valid),
			shuffle=True, callbacks=callbacks_list)
		all_predictions = per_class_metric.all_predictions
		all_recalls = per_class_metric.all_recalls
		for i in range(nb_epoch):
			print("this is the %d epoch" % i)
			print(all_predictions[i])
			print(all_recalls[i])
	else:
		print('Using real-time data augmentation.')
		# This will do preprocessing and realtime data augmentation:
		datagen = ImageDataGenerator(
		featurewise_center=False,  # set input mean to 0 over the dataset
		samplewise_center=False,  # set each sample mean to 0
		featurewise_std_normalization=False,  # divide inputs by std of the dataset
		samplewise_std_normalization=False,  # divide each input by its std
		zca_whitening=False,  # apply ZCA whitening
		rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
		width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
		height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
		horizontal_flip=True,  # randomly flip images
		vertical_flip=False)  # randomly flip images

		# Compute quantities required for featurewise normalization
		# (std, mean, and principal components if ZCA whitening is applied).
		datagen.fit(X_train)

		# Fit the model on the batches generated by datagen.flow().
		model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),
			samples_per_epoch=X_train.shape[0],
			validation_data=(X_valid, Y_valid),
			nb_epoch=nb_epoch, verbose=2, max_q_size=1000)

if __name__ == '__main__':
	if len(sys.argv) != 6:
		print('USAGE: python train.py <data_dir> <model_name> <pretrained> <optimizer_name> <weight_path>')
	else:
		data_dir = sys.argv[1]
		model_name = sys.argv[2]
		pretrained = sys.argv[3]
		optimizer_name = sys.argv[4]
		weight_path = sys.argv[5]
		train(data_dir, model_name, int(pretrained), optimizer_name, weight_path)
