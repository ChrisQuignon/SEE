#!/usr/bin/env python
import numpy as np
import cv2
import glob

#based on http://stackoverflow.com/questions/25233198/opencv-2-4-9-for-python-cannot-find-chessboard-camera-calibration-tutorial

# Arrays to store object points and image points from all the images.
imgpoints = []
images = glob.glob('img/*.tif')

for fname in images:
    ret = False
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #image must be scaled to gray, not only grayscale

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (12,13))

    print 'Chessboard on image ', fname , ": ", ret

    # If found, add object points, image points (after refining them)
    if ret == True:
        #stopping criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        cv2.cornerSubPix(gray, corners, (12,13), (-1,-1), criteria)
        imgpoints.append(corners)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, (12,13), corners, ret)

        #save or show image
        fname = fname.split('.')
        # cv2.imshow('img',img)
        cv2.imwrite(fname[0] + "_corners.jpg", img)
        # cv2.waitKey(0)
cv2.destroyAllWindows()
