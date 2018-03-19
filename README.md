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

- On peut utiliser https://en.wikipedia.org/wiki/Tesseract_(software) / https://github.com/tesseract-ocr/tesseract
Fonctionnement expliqué ici: https://static.googleusercontent.com/media/research.google.com/fr//pubs/archive/33418.pdf

- Un problème similaire: la lecture de plaques d'immatriculatio:
Avec un LSTM: https://hackernoon.com/latest-deep-learning-ocr-with-keras-and-supervisely-in-15-minutes-34aecd630ed8
Et un résau convolutionnel: https://towardsdatascience.com/number-plate-detection-with-supervisely-and-tensorflow-part-1-e84c74d4382c

- Un github avec plein d'info sur l'OCR: https://github.com/hs105/Deep-Learning-for-OCR
- Avec un réseau convolutionnel que l'on peut adapter: https://nicholastsmith.wordpress.com/2017/10/14/deep-learning-ocr-using-tensorflow-and-python/
- Autre idée: adapter le TP 2 (MNIST)

-
### Mesures d'erreur


# Bibliographie
Racing Bib Number Recognition, paper de Tel Aviv University, 2011

  - Appliquer directement un réseau de neurones sur nos photos pour repérer les dossards est inneficace: il y a trop de bruit dans l'environnement pour que le réseau les isole avec assurance
  - L'article montre qu'en isolant un individu avant de repérer son dossard on obtient des résultats bien plus statisfaisants
  - Leurs étapes:
    - Ils repèrent les visages puis distinguent une située 0.5*H en dessous de la tête (avec H la hauteur de la case contenant la tête) et dessinent un rectanglede taille 3*H x 7/3L (avec L la largeur de la box contenant la face)
    - Utilisation d'une SWT pour repérer les tags dans ces zones
    - Une ou deux transformations pour vérifier que le tag est bien entier
    - Utilisation de "Tesseract engine" pour lire le texte
