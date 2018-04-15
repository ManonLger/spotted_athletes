<h1 align='center'> Spotted Athletes </h1>
<p align='center'>
<i>Option Informatique - CentraleSupélec <br>
2017 - 2018 <hr></i></p>

__Auteur__ : Manon Léger, Emmanuel de Revel, Eymard Houdeville<br>

## Index
1. [Description](#description)

## <a name="description"></a>1. Description du problème
Ce projet a pour objectif de d'automatiser la reconnaissance de dossards dans les événements sportifs de l'école.
Notre problème peut donc être associé à un problème plus vaste où il s'agit de lire un document dans une photo de la vie réelle (plaque d'immatricuation,publicité, captcha...)

Il nous faut repérer entre 0 et n participants par image.

### Vérité terrain

Nous avons constitué un dataset avec:
- Des images trouvées sur internet
- Des images du Raid de Centrale

Nous avons donc un fichier avec le nom de l'image <-> les numéros de dossard que l'on y trouve qui nous permet de juger de la qualité de nos prédictions

(A tester aussi: https://supervise.ly/)

###  Pistes

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

