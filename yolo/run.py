import cv2, os, subprocess, sys
import numpy as np
from matplotlib import pyplot as plt


def make_yolo():
	os.chdir('../darknet')
	subprocess.call('make')
	os.chdir('../yolo')

def run_detector(txt_path, weights, to_detect):
	# Process txt file
	txt_path = '../samples/'+txt_path
	with open(txt_path, 'r') as t:
		images = t.read().split('\n')
		for i in range(len(images)):
			if images[i] == '\n' or images[i] == '':
				del images[i]
			else:
				if images[i][0:3] != '../':
					images[i] = '../samples/'+images[i]
				if images[i][-2:-1] != '\n':
					images[i] += '\n'
	with open(txt_path, 'w') as t:
		t.writelines(images)

	# Run detector
	if to_detect == 0:
		if 'yolov3.weights' not in os.listdir(os.getcwd()):
			subprocess.call(['wget', 'https://pjreddie.com/media/files/yolov3.weights'])
		os.chdir('../darknet')
		subprocess.call(['./darknet', 'detector', 'txt', 'cfg/coco.data', 'cfg/yolov3.cfg', '../yolo/yolov3.weights', txt_path])
	elif to_detect == 1:
		os.chdir('../darknet')
		subprocess.call(['./darknet', 'detector', 'txt', '../yolo/training/nnd.data', '../yolo/bib.cfg', weights, txt_path])
	os.chdir('../yolo')

def parse_boxes(txt_path, to_detect):
	# Prepare files and directories
	boxes_path = '../samples/'+txt_path+'_boxes.txt'
	with open(boxes_path, 'r') as b:
		boxes = b.read().split('\n')
	dir_path = txt_path+'_results'
	try:
		os.mkdir(dir_path)
	except OSError:
		pass

	if to_detect == 0:
		string = 'person'
	elif to_detect == 1:
		string = 'bib'

	# Parse boxes, crop and save images
	for b in boxes[:-1]:
		img_path, nb, left, right, top, bottom = b.split()
		img_name = '%s/%s_%s_%s.png' % (dir_path, img_path[11:], string, nb)
		cropped = crop_image(img_path, int(left), int(right), int(top), int(bottom))

		#rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
		#plt.imshow(rgb, interpolation='nearest')
		#plt.show()
		
		print('Saving to '+img_name)
		cv2.imwrite(img_name, cropped)

	os.remove(boxes_path)

def crop_image(img_path, left, right, top, bottom):
	img = cv2.imread(img_path, 1)
	width = right-left
	height = bottom-top
	cropped = np.zeros([height,width,3], dtype=np.uint8)
	for i in range(height):
		for j in range(width):
			cropped[i,j] = img[i+top, j+left]
	return cropped


if __name__ == '__main__':
	args = sys.argv
	if len(args) != 4:
		print("Please enter 3 arguments: \n+ The id corresponding to the class you wish to detect: 0 for 'person', 1 for 'bib' \n+ The name of the .txt file containing your list of images (eg: samples.txt). \n[This file MUST be placed in /samples directory.] \nSecond argument should be the .weights file to use (eg: nnd_final.weights). \n[This file MUST be placed in /weights directory.]")
	else:
		weights = '../yolo/training/weights/'+args[3]
		make_yolo()
		run_detector(args[2], weights, int(args[1]))
		parse_boxes(args[2], int(args[1]))