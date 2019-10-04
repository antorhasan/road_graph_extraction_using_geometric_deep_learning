import cv2 

img = cv2.imread('./data/img/minneapolis_1_0.png',1)

cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()