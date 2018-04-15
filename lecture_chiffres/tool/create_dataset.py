import os
import lmdb # install lmdb by "pip install lmdb"
import numpy as np
import six


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
        if not os.path.exists(imagePath):
            print('%s does not exist' % imagePath)
            continue
        with open(imagePath, 'rb') as f:
            imageBin = f.read()
        if checkValid:
            if not checkImageIsValid(imageBin):
                print('%s is not a valid image' % imagePath)
                continue

        imageKey = 'image-%09d' % cnt
        labelKey = 'label-%09d' % cnt
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
    #img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
    #imgH, imgW = img.shape[0], img.shape[1]
    #if imgH * imgW == 0:
        #return False
    return True


if __name__ == '__main__':
    data=[
        "/Users/eymardhoudeville/Documents/Centrale/Info/visord/spotted_athletes/crnn/crnn_pytorch/data/demo/demo.png"]*10000
    real=["1"]*10000
    createDataset("/Users/eymardhoudeville/Documents/Centrale/Info/visord/spotted_athletes/crnn/crnn_pytorch/data/new_data", data, real, lexiconList=None, checkValid=True)