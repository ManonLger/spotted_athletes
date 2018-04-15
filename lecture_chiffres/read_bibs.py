import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import pytesseract
import cv2
import os

# Options
WITH_TESSERACT = True

class read_with_crnn():
    def __init__(self):

    def predict(self):


class read_with_tesseract():
    def __init__(self,path):
        self.img = cv2.imread(path, 1)

    def predict(self):
        pred = pytesseract.image_to_string(Image.open(path))
        return pred

if __main__:
    if WITH_TESSERACT:
        print("Using tesseract")
        path = './samples/bib/bib1.png'
        R = read_with_tesseract(path)
        p=r.predict
        print(p)

    else:
        print("Using CRNN module")

