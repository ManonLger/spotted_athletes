import os
import lmdb # install lmdb by "pip install lmdb"
import numpy as np
import six
from os import listdir
from os.path import isfile, join
import cv2
from PIL import Image

def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)

def createDataset(outputPath, imagePathList, labelList, lexiconList=None, checkValid=False):
    """
    Create LMDB dataset for CRNN training.
    ARGS:
        outputPath    : LMDB output path
        imagePathList : list of image path
        labelList     : list of corresponding groundtruth texts
        lexiconList   : (optional) list of lexicon lists
        checkValid    : if true, check the validity of every image
    """

    assert(len(imagePathList) == len(labelList))
    nSamples = len(imagePathList)
    env = lmdb.open(outputPath, map_size=1099511627776)
    cache = {}
    cnt = 1
    for i in range(nSamples):
        imagePath = imagePathList[i]
        label = labelList[i]

        img = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (100, 32))
        cv2.imshow('image', img)
        cv2.imwrite(imagePath, img)

        imageKey = 'image-%09d' % cnt
        labelKey = 'label-%09d' % cnt

        with open(imagePath, 'rb') as f:
            imageBin = f.read()

        cache[imageKey] = imageBin

        cache[labelKey] = label

        if lexiconList:
            lexiconKey = 'lexicon-%09d' % cnt
            cache[lexiconKey] = ' '.join(lexiconList[i])
        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
            print('Written %d / %d' % (cnt, nSamples))
        cnt += 1
    nSamples = cnt-1
    cache['num-samples'] = str(nSamples)
    writeCache(env, cache)
    print("Created dataset with "+str(nSamples)+"  samples")


def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    imageBuf = np.fromstring(imageBin, dtype=np.uint8)
    img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (100, 32))
    imgH, imgW = img.shape[0], img.shape[1]
    cv2.imwrite('test.png', img)
    if imgH * imgW == 0:
        return False
    return img


if __name__ == '__main__':
    mypath="/Users/eymardhoudeville/Documents/Centrale/Info/visord/spotted_athletes/lecture_chiffres/training/data/English/chiffres"
    data=[]
    reality=[]
    for folder in listdir(mypath):
        if folder!='.DS_Store':
            print(folder)
            if folder!="Sample010":
                curr=int(folder[-1]) - 1
            else:
                curr=9
            print(curr)
            for file in listdir(mypath+"/"+folder):
                data.append(mypath+"/"+folder+"/"+file)
                reality.append(str(curr))

    createDataset("/Users/eymardhoudeville/Documents/Centrale/Info/visord/spotted_athletes/lecture_chiffres/training/data_training", data, reality, lexiconList=None, checkValid=True)