import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pytesseract
import os
import torch
from torch.autograd import Variable
import utils
import dataset
from PIL import Image
import os
import models.crnn as crnn
import numpy as np
import cv2

# Options
WITH_TESSERACT = True

class read_with_crnn():

    def __init__(self,path):
        self.model_path = './models/crnn.pth'
        self.img_path = path
        self.alphabet = '0123456789abcdefghijklmnopqrstuvwxyz'
        self.disp=True

    def predict(self):

        model = crnn.CRNN(32, 1, len(self.alphabet) + 1, 256)

        if self.disp:
            print('loading pretrained model from %s' % self.model_path)

        model.load_state_dict(torch.load(self.model_path))
        converter = utils.strLabelConverter(self.alphabet)
        transformer = dataset.resizeNormalize((100, 32))
        image = Image.open(self.img_path).convert('L')
        image = transformer(image)
        image = image.view(1, *image.size())
        image = Variable(image)
        model.eval()
        preds = model(image)

        _, preds = preds.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)

        preds_size = Variable(torch.IntTensor([preds.size(0)]))
        raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
        sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
        print('%-20s => %-20s' % (raw_pred, sim_pred))

        return sim_pred

class read_with_tesseract():
    def __init__(self,path):
        self.img = cv2.imread(path, 1)

    def predict(self):
        pred = pytesseract.image_to_string(Image.open(path))
        return pred

if __name__=='__main__':
    if WITH_TESSERACT:
        print("Using tesseract")
        path = './bib_samples/bib2.png'
        R = read_with_tesseract(path)
        p=R.predict()
        print(p)

    else:
        print("Using CRNN module")
        path='./bib_samples/bib2.png'
        C = read_with_crnn(path)
        c = C.predict()
        print(c)

