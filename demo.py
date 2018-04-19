import os

from yolo.detecting.Detector import Detector
from lecture_chiffres.read_bibs import read_with_crnn
from detection_chiffres.detection import Dossard

TXT_FILE = 'test.txt'

# ==============================
# == Detect person in picture ==
# ==============================

print('Using Yolo to detect person')
d = Detector(TXT_FILE)
d.run()

# ==============================
# == Detect bib in picture =====
# ==============================

print('Using Yolo to detect bib')
lines = []
for pers in os.listdir('yolo/detecting/'+TXT_FILE+'_results/person'):
	lines.append('../detecting/'+TXT_FILE+'_results/person/'+pers+'\n')
with open('yolo/detecting/people.txt', 'w') as f:
	f.writelines(lines)
d = Detector('people.txt', to_detect=1)
d.run()

# ==============================
# == Detect number on bib ======
# ==============================

print("Using Number detection module")
for bib_name in os.listdir('yolo/people.txt_results/bib'):
	dossard = Dossard(os.path.join('yolo/people.txt_results/bib', bib_name))
	number = dossard.numberDetection()
	cv2.imwrite("detection_chiffres/detection_outputs/"+ bib_name+ "_number_detected.jpg", number)
	print("Number detected")
# path = 'samples/bib/bib1.png'
# dossard=Dossard(path)
# number=dossard.numberDetection()
# cv2.imwrite("detection_chiffres/detection_outputs/"+ 'bib1'+ "_number_detected.jpg", number)
# print("Number detected")

# ==============================
# == Read a bib number =========
# ==============================

print("Using CRNN module")
for bib_name in os.listdir('detection_chiffres/detection_outputs'):
	C = read_with_crnn('detection_chiffres/detection_outputs'+bib_name)
	c = C.predict()
	print(c)
# path = 'samples/bib/69.png'
# C = read_with_crnn(path)
# c = C.predict()
# print(c)