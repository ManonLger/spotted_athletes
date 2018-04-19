import cv2, os, subprocess, sys
import numpy as np
# from matplotlib import pyplot as plt


class WrongClassToDetectException(Exception):
	pass

class Detector:
	"""
	:txt_path: name of the .txt file containing your list of images (eg: samples.txt).
	[This file MUST be placed in current directory.]
	:prefix: just a helper, if you wish to add a prefix to all the images paths in you .txt file
	:to_detect: class you wish to detect: 0 for 'person', 1 for 'bib'.
	:weights: name of the .weights file to use (eg: bib_final.weights).
	[This file MUST be placed in yolo/training/weights/ directory.]
	"""

	def __init__(self, txt_path, prefix=None, to_detect=0, weights=None):
		os.chdir('yolo/detecting')
		if to_detect == 0:
			self.to_detect = 'person'
		elif to_detect == 1:
			self.to_detect = 'bib'
		else:
			raise WrongClassToDetectException
		self.txt_path = txt_path
		if prefix:
			self._prefix_txt_file(prefix)
		self.weights = self._get_weights(weights)
		self._make_yolo()

	def run(self):
		self._run_detector()
		self._parse_boxes()

	def _get_weights(self, weights):
		if self.to_detect == 'person' and weights == None:
			if 'yolov3.weights' not in os.listdir(os.getcwd()):
				subprocess.call(['wget', 'https://pjreddie.com/media/files/yolov3.weights'])
			return 'yolov3.weights'
		elif self.to_detect == 'bib':
			if weights == None:
				return '../training/weights/bib_final.weights'
			else:
				return '../training/weights/'+weights
		else:
			print("Wrong arguments.")

	def _make_yolo(self):
		subprocess.call('../darknet/make')

	def _prefix_txt_file(self, prefix):
		with open(self.txt_path, 'r') as t:
			images = t.read().split('\n')
			for i in range(len(images)):
				if images[i] == '\n' or images[i] == '':
					del images[i]
				else:
					if images[i][:len(prefix)] != prefix:
						images[i] = prefix+images[i]
					if images[i][-2:-1] != '\n':
						images[i] += '\n'
		with open(self.txt_path, 'w') as t:
			t.writelines(images)

	def _run_detector(self):
		# Run detector
		if self.to_detect == 'person':
			subprocess.call(['../darknet/darknet', 'detector', 'txt', 'cfg/coco.data', 'cfg/yolov3.cfg', self.weights, self.txt_path])
		elif self.to_detect == 'bib':
			subprocess.call(['../darknet/darknet', 'detector', 'txt', '../training/bib.data', '../detecting/bib.cfg', self.weights, self.txt_path])

	def _parse_boxes(self):
		boxes_path = self.txt_path+'_boxes.txt'
		try:
			# Prepare files and directories
			with open(boxes_path, 'r') as b:
				boxes = b.read().split('\n')
			dir_path = self.txt_path+'_results'
			try:
				os.mkdir(dir_path)
			except OSError:
				pass

			try:
				os.chdir(dir_path)
				os.mkdir(self.to_detect)
				os.chdir('..')
			except OSError:
				pass

			# Parse boxes, crop and save images
			for b in boxes[:-1]:
				img_path, nb, left, right, top, bottom = b.split()
				img_name = '%s/%s/%s_%s.png' % (dir_path, self.to_detect, img_path[11:], nb)
				cropped = self._crop_image(img_path, int(left), int(right), int(top), int(bottom))

				# rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
				# plt.imshow(rgb, interpolation='nearest')
				# plt.show()
				
				print('Saving to '+img_name)
				cv2.imwrite(img_name, cropped)

			os.remove(boxes_path)
		
		except IOError:
			print('Nothing detected!')

	def _crop_image(self, img_path, left, right, top, bottom):
		img = cv2.imread(img_path, 1)
		width, height = right-left, bottom-top
		try:
			cropped = np.zeros([height,width,3], dtype=np.uint8)
			for i in range(height):
				for j in range(width):
					cropped[i,j] = img[i+top, j+left]
			return cropped
		except ValueError:
			return None


if __name__ == '__main__':
	os.chdir('../..')
	# d = Detector('people.txt')
	d = Detector('test.txt', to_detect=1)
	d.run()