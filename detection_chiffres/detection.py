import cv2


class Dossard:
    def __init__(self, path):

        img = cv2.imread(path, 1)
        self._img = img
        self._process_img = img

    @property
    def img(self):
        return self._img


    # Redimensionnement de l'image
    def _resize(self, size):
        self._img=cv2.resize(self._img, size)
        self._process_img=cv2.resize(self._process_img, size)

    # Seuil sur les valeurs de gris
    def _threshold(self, value):
        gray = cv2.cvtColor(self._process_img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, value, 255, cv2.THRESH_BINARY_INV)
        self._process_img=thresh

    # Dilatation pour améliorer la détection
    def _dilate(self, iterations_nb):
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        dilated = cv2.dilate(self._process_img, kernel, iterations=iterations_nb)
        self._process_img = dilated


    # Découpage du numéro
    def _cropNumber(self):

        # Détection des contours
        _, contours, hierarchy = cv2.findContours(self._process_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # Hauteur/largeur min/max des boites pouvant correspondre à des chiffres
        h_min = 50
        h_max = 70
        w_min = 25
        w_max = 40

        # On cherche les ordonnées y_min et y_max qui encadrent le mieux le numéro du dossard
        y_min = self._img.shape[0]
        y_max = 0

        for contour in contours:

            # chaque contour est associé à une boite
            [x, y, w, h] = cv2.boundingRect(contour)

            # on ne s'interesse qu'aux boites qui peuvent encadrer des chiffres (d'après hauteur/largeur min/max)
            if h > h_max and w > w_max:
                continue
            if h < h_min or w < w_min:
                continue

            if y + h > y_max:
                y_max = y + h
            if y < y_min:
                y_min = y

        # on découpe le dossard sur une bande entre y_min et y_max contenant le numéro
        self._img = self._img[y_min:y_max]

    #Ensemble du pipeline
    def numberDetection(self):
        self._resize((251, 243))
        self._threshold(130)
        self._dilate(1)
        self._cropNumber()


