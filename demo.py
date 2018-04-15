from lecture_chiffres.read_bibs import read_with_crnn
from detection_chiffres.detection import Dossard
from matplotlib import pyplot as plt


#Detect number on bib
print("Using Number detection module")
dossard=Dossard("dossard_raid_2.jpg")
number=dossard.numberDetection()
plt.imshow(number)
plt.show()
print("Number detected")


# Read a bib number
print("Using CRNN module")
path = 'samples/bib/bib1.png'
C = read_with_crnn(path)
c = C.predict()
print(c)
