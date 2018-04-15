from lecture_chiffres.read_bibs import read_with_crnn
from detection_chiffres.detection import Dossard


#Detect number on bib
print("Using Number detection module")
for bib_name in os.listdir("detection_chiffres/detection_inputs"):
    dossard=Dossard(os.path.join("detection_chiffres/detection_inputs", bib_name))
    number=dossard.numberDetection()
    cv2.imwrite("detection_chiffres/detection_outputs/"+ bib_name+ "_number_detected.jpg", number)
    print("Number detected")


# Read a bib number
print("Using CRNN module")
path = 'samples/bib/69.png'
C = read_with_crnn(path)
c = C.predict()
print(c)
