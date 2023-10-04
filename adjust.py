import cv2
import pandas as pd
import numpy as np

def ADJUST(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        colorsBGR = [x, y]
        print(colorsBGR)
        

cv2.namedWindow('Footage')
cv2.setMouseCallback('Footage', ADJUST)

cap=cv2.VideoCapture(r"C:\Users\Alok\Desktop\BTP\People_counter\People_counter\example-3.mp4")

while True:    
    ret,frame = cap.read()
    if not ret:
        break
    cv2.imshow("Footage", frame)

    if cv2.waitKey(0)&0xFF==27:
        break

cap.release()
cv2.destroyAllWindows()

