import cv2, os, subprocess, sys
import numpy as np
# from matplotlib import pyplot as plt


class Detector:
	"""
	:txt_path: name of the .txt file containing your list of images (eg: samples.txt).
	[This file MUST be placed in /samples directory.]
	:to_detect: class you wish to detect: 0 for 'person', 1 for 'bib'.
	:weights: name of the .weights file to use (eg: nnd_final.weights).
	[This file MUST be placed in /weights directory.]
	"""

	def __init__(txt_path, to_detect=0, weights=None):
		self._to_detect = to_detect
		self._txt_path = txt_path
		self._weights = self._get_weights(weights)
		self._make_yolo()

	def run():
		self._run_detector()
		self._parse_boxes()

	def _get_weights(weights):
		if self._to_detect == 0 and weights == None:
			if 'yolov3.weights' not in os.listdir(os.getcwd()):
				subprocess.call(['wget', 'https://pjreddie.com/media/files/yolov3.weights'])
			return '../yolo/yolov3.weights'
		elif self._to_detect == 1:
			if weights == None:
				return '../yolo/training/weights/nnd_final.weights'
			else:
				return '../yolo/training/weights/'+weights
		else:
			print("Wrong arguments.")

	def _make_yolo():
		os.chdir('../darknet')
		subprocess.call('make')
		os.chdir('../yolo')

	def _run_detector():
		# Process txt file
		path = '../samples/'+self._txt_path
		with open(path, 'r') as t:
			images = t.read().split('\n')
			for i in range(len(images)):
				if images[i] == '\n' or images[i] == '':
					del images[i]
				else:
					if images[i][0:3] != '../':
						images[i] = '../samples/'+images[i]
					if images[i][-2:-1] != '\n':
						images[i] += '\n'
		with open(path, 'w') as t:
			t.writelines(images)

		# Run detector
		os.chdir('../darknet')
		if self._to_detect == 0:
			subprocess.call(['./darknet', 'detector', 'txt', 'cfg/coco.data', 'cfg/yolov3.cfg', self._weights, path])
		elif self._to_detect == 1:
			subprocess.call(['./darknet', 'detector', 'txt', '../yolo/training/nnd.data', '../yolo/bib.cfg', self._weights, path])
		os.chdir('../yolo')

	def _parse_boxes():
		boxes_path = '../samples/'+self._txt_path+'_boxes.txt'
		try:
			# Prepare files and directories
			with open(boxes_path, 'r') as b:
				boxes = b.read().split('\n')
			dir_path = self._txt_path+'_results'
			try:
				os.mkdir(dir_path)
			except OSError:
				pass

			if self._to_detect == 0:
				class_name = 'person'
			elif self._to_detect == 1:
				class_name = 'bib'

			# Parse boxes, crop and save images
			for b in boxes[:-1]:
				img_path, nb, left, right, top, bottom = b.split()
				img_name = '%s/%s/%s_%s.png' % (dir_path, class_name, img_path[11:], nb)
				cropped = self._crop_image(img_path, int(left), int(right), int(top), int(bottom))

				# rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
				# plt.imshow(rgb, interpolation='nearest')
				# plt.show()
				
				print('Saving to '+img_name)
				cv2.imwrite(img_name, cropped)

			os.remove(boxes_path)
		
		except IOError:
			print('Nothing detected!')

	def _crop_image(img_path, left, right, top, bottom):
		img = cv2.imread(img_path, 1)
		width = right-left
		height = bottom-top
		cropped = np.zeros([height,width,3], dtype=np.uint8)
		for i in range(height):
			for j in range(width):
				cropped[i,j] = img[i+top, j+left]
		return cropped


if __name__ == '__main__':
	d = Detector('samples.txt', 0 'nnd_final.weights')
	# d = Detector('samples.txt', 1, 'nnd_final.weights')
	d.run()