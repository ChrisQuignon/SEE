#!/usr/bin/env python

from cv2 import *
cam = VideoCapture(0)   # 0 -> index of camera

cam.set(cv.CV_CAP_PROP_FRAME_WIDTH, 1920)
cam.set(cv.CV_CAP_PROP_FRAME_HEIGHT, 1080)
cam.set(cv.CV_CAP_PROP_BRIGHTNESS, 0.4)

i = 0

print 'press s to save an image'
print 'press q to quit'

while(True):
    # Capture frame-by-frame
    ret, frame = cam.read()
    if ret:
        # Our operations on the frame come here
        gray = cvtColor(frame, COLOR_BGR2GRAY)

        # Display the resulting frame
        imshow('frame',gray)
        if waitKey(113) & 0xFF == ord('q'):
            break
        if waitKey(155) & 0xFF == ord('s'):
            waitKey(1)
            imwrite("img/capture" + str(i) + ".jpg", gray)
            i = i + 1
            waitKey(1)
            # break
    else:
        break

# When everything done, release the capture
cam.release()
destroyAllWindows()
