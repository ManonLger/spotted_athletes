
from PIL import Image
import cv2

import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
import lecture_chiffres.models.crnn as crnn

import pytesseract

# Options
WITH_TESSERACT = False

class read_with_crnn():

    def __init__(self,path):
        self.model_path = 'lecture_chiffres/models/crnn.pth'
        self.img_path = path
        self.alphabet = '0123456789abcdefghijklmnopqrstuvwxyz'
        self.disp=True
        self.interpolation=Image.BILINEAR
        self.toTensor = transforms.ToTensor()
        self.size=(100, 32)

    def predict(self):

        model = crnn.CRNN(32, 1, len(self.alphabet) + 1, 256)

        if self.disp:
            print('loading pretrained model from %s' % self.model_path)

        model.load_state_dict(torch.load(self.model_path))

        self.alphabet = self.alphabet.lower()
        self.alphabet = self.alphabet + '-'
        self.dict = {}
        for i, char in enumerate(self.alphabet):
            self.dict[char] = i + 1

        image = Image.open(self.img_path).convert('L')

        image = image.resize(self.size, self.interpolation)
        image = self.toTensor(image)
        image.sub_(0.5).div_(0.5)

        image = image.view(1, *image.size())
        image = Variable(image)
        model.eval()
        preds = model(image)

        _, preds = preds.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)

        preds_size = Variable(torch.IntTensor([preds.size(0)]))
        raw_pred = self.decode(preds.data, preds_size.data, raw=True)
        sim_pred = self.decode(preds.data, preds_size.data, raw=False)
        print('%-20s => %-20s' % (raw_pred, sim_pred))

        return sim_pred

    def decode(self, t, length, raw=False):
        if length.numel() == 1:
            length = length[0]
            assert t.numel() == length, "text with length: {} does not match declared length: {}".format(t.numel(), length)
            if raw:
                return ''.join([self.alphabet[i - 1] for i in t])
            else:
                char_list = []
                for i in range(length):
                    if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                        char_list.append(self.alphabet[t[i] - 1])
                return ''.join(char_list)
        else:
            assert t.numel() == length.sum(), "texts with length: {} does not match declared length: {}".format(t.numel(), length.sum())
            texts = []
            index = 0
            for i in range(length.numel()):
                l = length[i]
                texts.append(
                    self.decode(
                        t[index:index + l], torch.IntTensor([l]), raw=raw))
                index += l
            return texts


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

