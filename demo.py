from lecture_chiffres.read_bibs import read_with_crnn
from detection_chiffres.detection import Dossard
from matplotlib import pyplot as plt


#Detect number on a bib
dossard=Dossard("dossard_raid_2.jpg")
dossard.numberDetection()
plt.imshow(dossard.img)
plt.show()

# Read a bib number
print("Using CRNN module")
path = 'samples/bib/bib1.png'
C = read_with_crnn(path)
c = C.predict()
print(c)
