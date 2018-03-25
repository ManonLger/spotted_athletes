import numpy as np
import cv2

img = cv2.imread('samples/bib/bib1.png',0)
print(img)

#cv2.imshow('image',img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(imgray,127,255,0)
im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img, contours, -1, (0,255,0), 3)

cnt = contours[4]
cv2.drawContours(img, [cnt], 0, (0,255,0), 3)
