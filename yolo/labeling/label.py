import cv2, os


if __name__ == '__main__':
	images = os.listdir('to_label')
	for img_name in images:
		img = cv2.imread('to_label/'+img_name, 1)
		small = cv2.resize(img, None, fx=0.15, fy=0.15, interpolation=cv2.INTER_CUBIC)
		cv2.imwrite('OpenLabeling/images/'+img_name, small)