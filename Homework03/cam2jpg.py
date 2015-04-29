#!/usr/bin/env python

from cv2 import *
cam = VideoCapture(1)   # 0 -> index of camera
s, img = cam.read()
if s: #no errors
    namedWindow("cam-test",CV_WINDOW_AUTOSIZE)
    print 'image taken'
    startWindowThread()
    imshow("cam-test",img)
    waitKey(1)
    imwrite("images/testimg.jpg", img)
    cam.release()
    destroyWindow("cam-test")
    waitKey(1)
