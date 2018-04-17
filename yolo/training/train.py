import os, subprocess
from random import randrange


def get_weights():
	if 'darknet53.conv.74' not in os.listdir(os.getcwd()):
		subprocess.call(['wget', 'https://pjreddie.com/media/files/darknet53.conv.74'])

def generate_sets():
	images = [ img for img in os.listdir('../labeling/labeled_img') if img[-4:] != '.txt' ]
	train_set, val_set, test_set = [], [], []

	# Randomly generate sets from available images
	n = len(images)
	def get_r():
		r = randrange(n)
		while any((r in train_set, r in val_set, r in test_set)):
			r = randrange(n)
		return r

	for i in range(200):
		train_set.append(get_r())
	for i in range(80):
		val_set.append(get_r())
	for i in range(20):
		test_set.append(get_r())

	def get_path(img):
		return os.getcwd()[:-8]+'labeling/labeled_img/'+img+'\n'

	train_set = [ get_path(images[i]) for i in train_set ]
	val_set = [ get_path(images[i]) for i in val_set ]
	test_set = [ get_path(images[i]) for i in test_set ]

	# Write sets to files
	with open('train.txt', 'w') as f:
		f.writelines(train_set)
	with open('val.txt', 'w') as f:
		f.writelines(val_set)
	with open('test.txt', 'w') as f:
		f.writelines(test_set)

def run_training():
	os.chdir('../../darknet')
	subprocess.call(['./darknet', 'detector', 'train', '../yolo/training/nnd.data', '../yolo/training/nnd.cfg', '../yolo/training/weights/nnd_700.weights'])


if __name__ == '__main__':
	get_weights()
	generate_sets()
	run_training()
