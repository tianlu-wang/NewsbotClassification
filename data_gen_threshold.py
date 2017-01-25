import json
import sys, os, random
from PIL import Image
import numpy as np 

def data_gen_threshold(images_json_file, images_dir, output_dir, threshold):
	images = json.load(open(images_json_file))
	random.shuffle(images)  # shuffle data 
	y_data = list()
	category2nb = {"politics_elections":0, "science_technology":1, "law_crime":2, "sports":3,
	"religion":4, "disaster_accident":5, "health_medicine_environment":6, "business_economy":7,
	"arts_culture":8, "international_relations":9, "conflict_attack":10, "education":11}
	counter = 0
	nb_images = 0
	for image in images:
		nb_images += 1
		try:
			if image['identify_ok'] == 1 and image["image_shape"][0] > 224 and image["image_shape"][1] > 224:
				topic = image['topic1']
				if image['topic_scores'][topic] > threshold:  # set the confidence threshold
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

	### save matrix to file
	x_file = open("%s/x_%s.data" % (output_dir, str(len(y_data))), "wb")
	y_file = open("%s/y_%s.data" % (output_dir, str(len(y_data))), "wb")
	
	np.save(x_file, x_data)
	np.save(y_file, np.array(y_data))

def img2array(images_dir, id):  # may need some modification according to 
	identifier = '%07d' % id
	image_path = '%s/%s' % (images_dir, identifier[:4])
	image_fullpath = '%s/%s.jpg' % (image_path, identifier[4:])

	with Image.open(image_fullpath) as im:
		im = im.resize((224, 224))
		array = np.asarray(im)

	return array

if __name__ == '__main__':
    if len(sys.argv) != 5:
        print 'USAGE: python data_gen_threshold.py <images_json_file> <images_dir> <output_dir> <threshold>'
        print 'generate data for classification task on Newsbot according to the threshold'
    else:
        images_json_file = sys.argv[1]
        images_dir = sys.argv[2]
        output_dir = sys.argv[3]
        threshold = float(sys.argv[4])
        data_gen_threshold(images_json_file, images_dir, output_dir, threshold)
