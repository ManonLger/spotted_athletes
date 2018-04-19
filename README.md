<h1 align='center'> Spotted Athletes </h1>
<p align='center'>
<i>Option Informatique - CentraleSupélec <br>
2017 - 2018 <hr></i></p>

__Auteur__ : Manon Léger, Emmanuel de Revel, Eymard Houdeville<br>

## Index
1. [Installation](#installation)
2. [Usage](#usage)
3. [Description](#description)

## <a name="installation"></a>1. Installation
**Pré-requis**
+ Environnement UNIX
+ Python 3.6
+ OpenCV 2

**Installation**
```
git clone https://gitlab.centralesupelec.fr/2014houdevile/spotted_athletes.git
cd spotted_athletes/yolo/darknet
git clone https://github.com/Manon-L/darknet.git
cd ../labeling/OpenLabeling
git clone https://github.com/Cartucho/OpenLabeling.git
cd ../../..
```

## <a name="usage"></a>2. Usage
Pour détecter les numéros de dossard sur l'ensemble de test, il suffit d'exécuter `demo.py`.
Pour ré-entraîner le modèle, exécuter `Trainer.py`.

## <a name="description"></a>3. Description du problème
Ce projet a pour objectif de d'automatiser la reconnaissance de dossards dans les événements sportifs de l'école.
Notre problème peut donc être associé à un problème plus vaste où il s'agit de lire un document dans une photo de la vie réelle (plaque d'immatricuation,publicité, captcha...)

Il nous faut repérer entre 0 et n participants par image.

### Vérité terrain

Nous avons constitué un dataset avec:
- Des images trouvées sur internet
- Des images du Night'N'Day de CentraleSupélec

Nous avons donc un fichier avec le nom de l'image <-> les numéros de dossard que l'on y trouve qui nous permet de juger de la qualité de nos prédictions

(A tester aussi: https://supervise.ly/)

### Pistes

#### Object detection
Comparaison des méthodes de deep learning pour la détection d'objets (ici coureur puis dossard) : http://cv-tricks.com/object-detection/faster-r-cnn-yolo-ssd/

#### The stroke width transform (SWT)
Repose sur l'idée que les dossards ont toujours en théorie la même forme et la même organisation.
SWT n'est pas implémenté encore dans openCV apparemment. Mais: https://github.com/subokita/Robust-Text-Detection

#### Lecture du texte (OCR)

- Reconnaissance des tickets de caisse au supermarché: https://dzone.com/articles/using-ocr-for-receipt-recognition

- On peut utiliser https://en.wikipedia.org/wiki/Tesseract_(software) / https://github.com/tesseract-ocr/tesseract
Fonctionnement expliqué ici: https://static.googleusercontent.com/media/research.google.com/fr//pubs/archive/33418.pdf

- Un problème similaire: la lecture de plaques d'immatriculatio:
Avec un LSTM: https://hackernoon.com/latest-deep-learning-ocr-with-keras-and-supervisely-in-15-minutes-34aecd630ed8
Et un résau convolutionnel: https://towardsdatascience.com/number-plate-detection-with-supervisely-and-tensorflow-part-1-e84c74d4382c

- Un github avec plein d'info sur l'OCR: https://github.com/hs105/Deep-Learning-for-OCR
- Avec un réseau convolutionnel que l'on peut adapter: https://nicholastsmith.wordpress.com/2017/10/14/deep-learning-ocr-using-tensorflow-and-python/
- Autre idée: adapter le TP 2 (MNIST)

