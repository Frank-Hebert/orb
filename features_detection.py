import cv2
import numpy as np


img = cv2.imread("duckiebot_1.png", cv2.IMREAD_GRAYSCALE)



orb = cv2.ORB_create(nfeatures=15000)


keypoints_orb, descriptors = orb.detectAndCompute(img, None)

img = cv2.drawKeypoints(img, keypoints_orb, None)
height, width = img.shape[:2]

cv2.namedWindow('jpg', cv2.WINDOW_NORMAL)
cv2.resizeWindow('jpg', width, height)
cv2.imshow('jpg', img)
cv2.waitKey(0)

cv2.destroyAllWindows()






# cv2.imshow("Image", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()