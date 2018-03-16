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

###  Pistes

#### The stroke width transform (SWT)
Repose sur l'idée que les dossards ont toujours en théorie la même forme et la même organisation.
SWT n'est pas implémenté encore dans openCV apparemment. Mais: https://github.com/subokita/Robust-Text-Detection

#### Lecture du texte (OCR)

- On peut utiliser https://en.wikipedia.org/wiki/Tesseract_(software) / https://github.com/tesseract-ocr/tesseract
Fonctionnement expliqué ici: https://static.googleusercontent.com/media/research.google.com/fr//pubs/archive/33418.pdf


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
