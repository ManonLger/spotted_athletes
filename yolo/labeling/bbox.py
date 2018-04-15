import cv2, os, shutil


if __name__ == '__main__':
	os.chdir('OpenLabeling')
	txts = [ txt[:-4] for txt in os.listdir('bbox_txt') ]
	images = os.listdir('images')
	for img_name in images:
		if img_name[:-4] in txts:
			shutil.copy('images/'+img_name, '../labeled_img/'+img_name)