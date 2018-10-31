import numpy as np
import cv2
faceDetect=cv2.CascadeClassifier('haarcascade_frontface_default.xml')
cam=cv2.VideoCapture(-1)
while cam.isOpened():
    ret,img=cam.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=faceDetect.detectMultiscale(gray,scalefactor=1.3,minneighbours=5,minsize=(30,30))
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+y,y+h),(0,0,255),2)
    cv2.imshow("Faces",img)
    if(cv2.waitkey(1)==ord('q')):
        break
cam.release()
cv2.destroyAllWindows()
    


