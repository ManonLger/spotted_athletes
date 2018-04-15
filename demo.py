from yolo import Detector
from lecture_chiffres.read_bibs import read_with_crnn
from detection_chiffres.detection import Dossard

TXT_FILE = 'samples.txt'

# ==== Detect person in picture ====
print('Using Yolo to detect person')
	d = Detector(TXT_FILE)
	d.run()

# ==== Detect bib in picture ====

# print('Using Yolo to detect bib')
# 	d = Detector(TXT_FILE, 1, 'nnd_final.weights')
# 	d.run()

# ==== Detect number on bib ====

print("Using Number detection module")
for bib_name in os.listdir('yolo/'+TXT_FILE+'_results/bib'):
    dossard=Dossard(os.path.join('yolo/'+TXT_FILE+'_results/bib', bib_name))
    number=dossard.numberDetection()
    cv2.imwrite("detection_chiffres/detection_outputs/"+ bib_name+ "_number_detected.jpg", number)
    print("Number detected")

# ==== Read a bib number ====

print("Using CRNN module")
# for bib_name in os.listdir('detection_chiffres/detection_outputs'):
# 	C = read_with_crnn('detection_chiffres/detection_outputs'+bib_name)
# 	c = C.predict()
# 	print(c)
path = 'samples/bib/69.png'
C = read_with_crnn(path)
c = C.predict()
print(c)