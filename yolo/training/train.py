import os, subprocess
from random import randrange


def generate_sets():
	images = os.listdir('../labeling/labeled_img')
	train_set, val_set, test_set = [], [], []

	# Randomly generate sets from available images
	n = len(images)
	for i in range(10):
		for s in (train_set, val_set, test_set):
			r = randrange(n)
			while any((r in train_set, r in val_set, r in test_set)):
				r = randrange(n)
			s.append(r)

	train_set = [ images[i]+'\n' for i in train_set ]
	val_set = [ images[i]+'\n' for i in val_set ]
	test_set = [ images[i]+'\n' for i in test_set ]

	# Write sets to files
	with open('train.txt', 'w') as f:
		f.writelines(train_set)
	with open('val.txt', 'w') as f:
		f.writelines(val_set)
	with open('test.txt', 'w') as f:
		f.writelines(test_set)

def run_training():
	os.chdir('../../darknet')
	subprocess.call(['./darknet', 'detector', 'train', '../yolo/training/nnd.data', 'cfg/yolov3-voc.cfg', '../yolo/training/darknet53.conv.74'])
	os.chdir('../training/yolo')


if __name__ == '__main__':
	generate_sets()
	run_training()
