import cv2
import numpy as np
from Util import Util

util = Util()
cam = cv2.VideoCapture(0)
neutral_angle = 0
while cam.isOpened():
    ret, frame = cam.read()
    
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)[:, ::-1]
        
    util.add_effect(frame, 'effects/example1.png')
    cv2.imshow("effect", frame)
    if cv2.waitKey(1) & 0xff == 27:
        cv2.destroyWindow("effect")
        cam.release()
        break
    