import cv2, os
# from matplotlib import pyplot as plt


if __name__ == '__main__':
	images = os.listdir('nnd')
	for img_name in images:
		img = cv2.imread('nnd/'+img_name, 1)
		small = cv2.resize(img, None, fx=0.15, fy=0.15, interpolation = cv2.INTER_CUBIC)
		cv2.imwrite('OpenLabeling/images/'+img_name, small)