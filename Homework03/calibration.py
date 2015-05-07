#!/usr/bin/env python
import numpy as np
import cv2
import glob

#based on http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html

images = glob.glob('img/capture/*.jpg')


#we stop after either 30 iterations or if the precision of 0.001 is met
stop = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

#dimensionality of the chessboard (inner cornerpoints)
chessdim = (8,8)

#3D object points as a grid with z = 0
objp = np.zeros((chessdim[0]*chessdim[1],3), np.float32)
objp[:,:2] = np.mgrid[0:chessdim[0],0:chessdim[1]].T.reshape(-1,2)



#2D image points
imgpoints = []

#3d object points
objpoints = []

for fname in images:
    ret = False


    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    #image must be scaled to gray, not only grayscale

    # Find the chess board corners
    #The numbers of inside corners of the chessboard
    ret, corners = cv2.findChessboardCorners(gray, chessdim)

    # If found, add object points, image points (after refining them)
    if ret == True:
        print 'Chessboard on image ', fname, ' found.'
        objpoints.append(objp)
        cv2.cornerSubPix(gray, corners, chessdim, (-1,-1), stop)
        imgpoints.append(corners)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, chessdim, corners, ret)

        #save or show image
        fname = fname.split('.')
        # cv2.imshow('img',img)
        cv2.imwrite(fname[0] + "_corners.jpg", img)
        # cv2.waitKey(0)


cv2.destroyAllWindows()


#calibration
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

print '\nret: \n', ret
print '\nmtx: \n', mtx
print '\ndist: \n', dist
print '\nrvecs: \n', rvecs
print '\ntvecs: \n', tvecs

img = cv2.imread('img/capture/cap_1.jpg')
h,  w = img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

print '\nnewcameramtx: \n', newcameramtx
print '\nroi: \n', roi
