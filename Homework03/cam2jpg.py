#!/usr/bin/env python

from cv2 import *
# initialize the camera
cam = VideoCapture(1)   # 0 -> index of camera
s, img = cam.read()
if s:    # frame captured without any errors
    namedWindow("cam-test",CV_WINDOW_AUTOSIZE)

    startWindowThread()
    imshow("cam-test",img)
    cam.release()
    waitKey(1)
    destroyWindow("cam-test")
    waitKey(1)
    imwrite("images/testimg.jpg",img) #save image

