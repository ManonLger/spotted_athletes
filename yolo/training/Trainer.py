import os, subprocess
from random import randrange


class Trainer:
	"""
	:train_set_ok: Set to true if you want to keep the train set already stored in train.txt
	:val_set_ok: Set to true if you want to keep the validation set already stored in val.txt
	:weights: name of the .weights file to use (eg: bib_final.weights).
	[This file MUST be placed in the yolo/training/weights/ directory.]
	"""

	def __init__(self, reshape_sets=False, weights=None):
		os.chdir('yolo/training')
		self.weights = _get_weights(weights)
		current_dir = os.listdir(os.getcwd())
		if reshape_sets or 'train.txt' not in current_dir or 'val.txt' not in current_dir or 'test.txt' not in current_dir:
			self._generate_sets()

	def train(self):
		subprocess.call(['../darknet/darknet', 'detector', 'train', '../training/bib.data', '../training/bib.cfg', '../training/'+self.weights])

	def _get_weights(self, weights):
		w = 'weights/'
		try:
			os.mkdir('weights')
		except OSError:
			pass

		if weights != None:
			return w+weights
		else:
			if 'darknet53.conv.74' not in os.listdir('weights'):
				os.chdir('weights')
				subprocess.call(['wget', 'https://pjreddie.com/media/files/darknet53.conv.74'])
				os.chdir('..')
			return w+'darknet53.conv.74'

	def _generate_sets(self):
		train_set, val_set, test_set = [], [], []
		images = [ img for img in os.listdir('../labeling/labeled_img') if img[-4:] != '.txt' ]

		# Randomly generate sets
		n = len(images)
		def get_r():
			r = randrange(n)
			while any((r in train_set, r in val_set)):
				r = randrange(n)
			return r

		for i in range(200):
			train_set.append(get_r())
		for i in range(100):
			val_set.append(get_r())
		test_set = range(len(images)) - set(train_set) - set(val_set)

		# Get corresponding images paths
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
		with open('../detecting/test.txt', 'w') as f:
			f.writelines(test_set)


if __name__ == '__main__':
	os.chdir('../..')
	t = Trainer(reshape_sets=True)
	t.train()