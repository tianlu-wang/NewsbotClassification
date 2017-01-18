import json
import sys, os, random
from PIL import Image
import numpy as np 

def data_gen(images_json_file, images_dir, output_dir, nb_train, nb_valid, nb_test):
	images = json.load(open(images_json_file))
	random.shuffle(images)  # shuffle data 
	y_data = list()
	category2nb = {"politics_elections":0, "science_technology":1, "law_crime":2, "sports":3,
	"religion":4, "disaster_accident":5, "health_medicine_environment":6, "business_economy":7,
	"arts_culture":8, "international_relations":9, "conflict_attack":10, "education":11}
	nb = nb_train + nb_valid + nb_test
	counter = 0
	nb_images = 0
	for image in images:
		nb_images += 1
		if counter >= nb:
			break
		try:
			if image['identify_ok'] == 1 and image["image_shape"][0] > 224 and image["image_shape"][1] > 224:
				topic = image['topic1']
				if image['topic_scores'][topic] > 0.45:  # set the confidence threshold
					img_array = img2array(images_dir, image['id'])
					if counter == 0:
						x_data = np.array([img_array])
					else:
						x_data = np.concatenate((x_data, np.array([img_array])), axis=0)
					y_data.append(category2nb[topic])
					counter += 1
				if counter % 100 == 0:
					print("counter now is %d" % counter)
					print("nb_images now is %d" % nb_images)
		except Exception as ex:
			template = "An exception of type {0} occured. Arguments:\n{1!r}"
			message = template.format(type(ex).__name__, ex.args)
			print message 
			print image["image_shape"]

	print("The shape of x data is: "),
	print(x_data.shape)
	print("The length of y data is: "),
	print(len(y_data))

	### divide into train, valid and test
	x_train = x_data[:nb_train, :, :, :]
	y_train = y_data[ :nb_train]

	print("The shape of x train data is: "),
	print(x_train.shape)
	print("The length of y train data is: "),
	print(len(y_train))

	x_valid = x_data[nb_train : nb_train+nb_valid, :, :, :]
	y_valid = y_data[nb_train : nb_train+nb_valid ]
	print("The shape of x valid data is: "),
	print(x_valid.shape)
	print("The length of y valid data is: "),
	print(len(y_valid))

	x_test = x_data[nb_train+nb_valid : , :, :, :]
	y_test = y_data[nb_train+nb_valid : ]
	print("The shape of x test data is: "),
	print(x_test.shape)
	print("The length of y test data is: "),
	print(len(y_test))

	### save matrix to file
	x_train_file = open("%s/x_train.data" % output_dir, "wb")
	y_train_file = open("%s/y_train.data" % output_dir, "wb")
	x_valid_file = open("%s/x_valid.data" % output_dir, "wb")
	y_valid_file = open("%s/y_valid.data" % output_dir, "wb")
	x_test_file = open("%s/x_test.data" % output_dir, "wb")
	y_test_file = open("%s/y_test.data" % output_dir, "wb")

	
	np.save(x_train_file, x_train)
	np.save(y_train_file, np.array(y_train))
	np.save(x_valid_file, x_valid)
	np.save(y_valid_file, np.array(y_valid))
	np.save(x_test_file, x_test)
	np.save(y_test_file, np.array(y_test))

def img2array(images_dir, id):
	identifier = '%07d' % id
	image_path = '%s/%s' % (images_dir, identifier[:4])
	image_fullpath = '%s/%s.jpg' % (image_path, identifier[4:])

	with Image.open(image_fullpath) as im:
		im = im.resize((224, 224))
		array = np.asarray(im)

	return array

if __name__ == '__main__':
    if len(sys.argv) != 7:
        print 'USAGE: python data_gen.py <images_json_file> <images_dir> <output_dir> <nb_train> <nb_valid> <nb_test>'
        print 'generate data for classification task on Newsbot'
    else:
        images_json_file = sys.argv[1]
        images_dir = sys.argv[2]
        output_dir = sys.argv[3]
        nb_train = int(sys.argv[4])
        nb_valid = int(sys.argv[5])
        nb_test = int(sys.argv[6])
        data_gen(images_json_file, images_dir, output_dir, nb_train, nb_valid, nb_test)
