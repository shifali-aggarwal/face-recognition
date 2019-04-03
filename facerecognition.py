import numpy as np
import cv2

face_cascade=cv2.CascadeClassifier('C:/Users/Innayat/Desktop/shifali/cascades/data/haarcascade_frontalface_alt2.xml')

cap=cv2.VideoCapture(0)
while True:
    ret,frame= cap.read()
    gray= cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray,scaleFactor=1.5,minNeighbors=5)
    for (x,y,w,h) in faces:
        print(x,y,w,h)
        roi_gray =gray[y:y+h,x:x+w]#refion of intrest actually where the photo of the image stores
        roi_color=frame[y:y+h,x:x+w]#gives color image
        img_item='image'+str(x)+str(y)+'.png'#saving the image
        cv2.imwrite(img_item,roi_color)
        #recognizing the person involves deeplearning pytorch keras tensorflow sckitlearn

                
        color=(255,0,0)#BGR not rbg
        stroke = 2
        end_corx=x+w
        end_cordy=y+h
        cv2.rectangle(frame,(x,y),(end_corx,end_cordy),color,stroke)
    cv2.imshow('frame',frame)
    if cv2.waitKey(20)&0xFF ==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
